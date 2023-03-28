import os, sys
import logging
from typing import Optional,Sequence
from pflow.common.gromacs.gmx_constant import gmx_trjconv_cmd, gmx_traj_cmd
from pflow.constants import gmx_align_name
from pflow.utils import list_to_string
from pflow.utils import run_command


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def pbc_trjconv(
        xtc: str,
        top: str = "topol.tpr",
        output_group: int = 0,
        output: str = "md_nopbc.xtc"
    ):
    logger.info("handling pbc by gmx trjconv command ...")
    cmd_list = gmx_trjconv_cmd.split()
    cmd_list += ["-f", str(xtc)]
    cmd_list += ["-s", str(top)]
    cmd_list += ["-o", output]
    cmd_list += ["-ur", "compact"]
    cmd_list += ["-pbc", "mol"]
    logger.info(list_to_string(cmd_list, " "))
    return_code, out, err = run_command(
        cmd_list,
        stdin=f"{output_group}\n"
    )
    assert return_code == 0, err
    
    
def align_trjconv(
        xtc: str = "md_nopbc.xtc",
        top: str = "topol.tpr",
        output_group: int = 1,
        output: str = gmx_align_name
    ):
    logger.info("handling pbc by gmx trjconv command ...")
    cmd_list = gmx_trjconv_cmd.split()
    cmd_list += ["-f", str(xtc)]
    cmd_list += ["-s", str(top)]
    cmd_list += ["-o", output]
    cmd_list += ["-fit", "rot+trans"]
    logger.info(list_to_string(cmd_list, " "))
    return_code, out, err = run_command(
        cmd_list,
        stdin=f"{1}\n{output_group}\n"
    )
    assert return_code == 0, err
    
def begin_trjconv(
        xtc: str = gmx_align_name,
        top: str = "topol.tpr",
        output_group: int = 1,
        output: str = "begin.gro"
    ):
    logger.info("generating first frame by gmx trjconv command ...")
    cmd_list = gmx_trjconv_cmd.split()
    cmd_list += ["-f", str(xtc)]
    cmd_list += ["-s", str(top)]
    cmd_list += ["-o", output]
    cmd_list += ["-b", "0"]
    cmd_list += ["-e", "0"]
    logger.info(list_to_string(cmd_list, " "))
    return_code, out, err = run_command(
        cmd_list,
        stdin=f"{output_group}\n"
    )
    assert return_code == 0, err

def slice_trjconv(
        xtc: str,
        top: str,
        selected_time: float,
        output_group: int = 0,
        output: str = "conf.gro"
    ):
    logger.info("slicing trajectories by gmx trjconv command ...")
    logger.warning("You are using `gmx trjconv` to slice trajectory, "
    "make sure that the selected index is in the unit of time (ps).")
    cmd_list = gmx_trjconv_cmd.split()
    cmd_list += ["-f", str(xtc)]
    cmd_list += ["-s", str(top)]
    cmd_list += ["-dump", str(selected_time)]
    cmd_list += ["-o", output]
    logger.info(list_to_string(cmd_list, " "))
    return_code, out, err = run_command(
        cmd_list,
        stdin=f"{output_group}\n"
    )
    assert return_code == 0, err
    

    