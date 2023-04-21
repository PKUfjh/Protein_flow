from typing import Dict, List
from copy import deepcopy
from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Step,
    Steps,
    argo_range,
    argo_len,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    Slices,
)
from pflow.utils import init_executor


class Label(Steps):
    
    r"""" Label SuperOP.
    This SuperOP combines CheckLabelInputs OP, PrepLabel OP and RunLabel OP.   
    """
    def __init__(
        self,
        name: str,
        check_input_op: OP,
        prep_op: OP,
        run_op: OP,
        prep_config: Dict,
        run_config: Dict,
        upload_python_package = None,
        retry_times = None
    ):

        self._input_parameters = {
            "label_config": InputParameter(type=Dict),
            "label_cv_config": InputParameter(type=Dict)
        }        
        self._input_artifacts = {
            "topology" : InputArtifact(optional=True),
            "forcefield" : InputArtifact(optional=True),
            "confs": InputArtifact(),
            "index_file": InputArtifact(optional=True),
            "conf_tags": InputArtifact(optional=True)
        }
        self._output_parameters = {
            "conf_tags": OutputParameter(type=List)
        }
        self._output_artifacts = {
            "conf_begin": OutputArtifact(),
            "trajectory_aligned": OutputArtifact()
        }

        super().__init__(        
                name=name,
                inputs=Inputs(
                    parameters=self._input_parameters,
                    artifacts=self._input_artifacts
                ),
                outputs=Outputs(
                    parameters=self._output_parameters,
                    artifacts=self._output_artifacts
                ),
            )
        
        step_keys = {
            "check_label_inputs": "check-label-inputs",
            "prep_label": "prep-label",
            "run_label": "run-label",
        }

        self = _label(
            self, 
            step_keys,
            check_input_op,
            prep_op,
            run_op,
            prep_config = prep_config,
            run_config = run_config,
            upload_python_package = upload_python_package,
            retry_times = retry_times
        )            
    
    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _label(
        label_steps,
        step_keys,
        check_label_input_op : OP,
        prep_label_op : OP,
        run_label_op : OP,
        prep_config : Dict,
        run_config : Dict,
        upload_python_package : str = None,
        retry_times: int = None
    ):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)

    prep_template_config = prep_config.pop('template_config')
    run_template_config = run_config.pop('template_config')

    prep_executor = init_executor(prep_config.pop('executor'))
    run_executor = init_executor(run_config.pop('executor'))

    check_label_inputs = Step(
        'check-label-inputs',
        template=PythonOPTemplate(
            check_label_input_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            **prep_template_config,
        ),
        parameters={},
        artifacts={
            "conf_tags": label_steps.inputs.artifacts['conf_tags'],  
            "confs": label_steps.inputs.artifacts['confs'],
        },
        key = step_keys['check_label_inputs'],
        executor = prep_executor,
        **prep_config,
    )
    label_steps.add(check_label_inputs)

    prep_label = Step(
        'prep-label',
        template=PythonOPTemplate(
            prep_label_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices("{{item}}",
                group_size=200,
                pool_size=1,
                input_parameter=["task_name"],
                input_artifact=["conf"],
                output_artifact=["task_path"]),
            **prep_template_config,
        ),
        parameters={
            "label_config": label_steps.inputs.parameters['label_config'],
            "label_cv_config": label_steps.inputs.parameters['label_cv_config'],
            "task_name": check_label_inputs.outputs.parameters['conf_tags']
        },
        artifacts={
            "topology": label_steps.inputs.artifacts['topology'],
            "conf": label_steps.inputs.artifacts['confs']
        },
        key = step_keys['prep_label']+"-{{item}}",
        executor = prep_executor,
        with_param=argo_range(argo_len(check_label_inputs.outputs.parameters['conf_tags'])),
        when = "%s > 0" % (check_label_inputs.outputs.parameters["if_continue"]),
        **prep_config,
    )
    label_steps.add(prep_label)

    run_label = Step(
        'run-label',
        template=PythonOPTemplate(
            run_label_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices("{{item}}",
                group_size=200,
                pool_size=1,
                input_parameter=["task_name"],
                input_artifact=["task_path"],
                output_artifact=["plm_out","plm_fig","trajectory_aligned","conf_begin","md_log"]),
            **run_template_config,
        ),
        parameters={
            "label_config": label_steps.inputs.parameters["label_config"],
            "label_cv_config": label_steps.inputs.parameters['label_cv_config'],
            "task_name": check_label_inputs.outputs.parameters['conf_tags'],
        },
        artifacts={
            "forcefield": label_steps.inputs.artifacts['forcefield'],
            "task_path": prep_label.outputs.artifacts["task_path"],
            "index_file": label_steps.inputs.artifacts['index_file'],
        },
        key = step_keys['run_label']+"-{{item}}",
        executor = run_executor,
        with_param=argo_range(argo_len(check_label_inputs.outputs.parameters['conf_tags'])),
        continue_on_success_ratio = 0.75,
        **run_config,
    )
    label_steps.add(run_label)

    label_steps.outputs.parameters["conf_tags"].value_from_parameter = check_label_inputs.outputs.parameters["conf_tags"]
    label_steps.outputs.artifacts["trajectory_aligned"]._from = run_label.outputs.artifacts["trajectory_aligned"]
    label_steps.outputs.artifacts["conf_begin"]._from = run_label.outputs.artifacts["conf_begin"]
    
    return label_steps
