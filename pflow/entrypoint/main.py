import argparse, os, glob, sys, logging
from operator import index
from pathlib import Path
from typing import (
    Optional,
    List,
)
import pflow

if os.getenv('DFLOW_DEBUG'):
    from dflow import config
    config["mode"] = "debug"
    
NUMEXPR_MAX_THREADS = os.getenv("NUMEXPR_MAX_THREADS")
if NUMEXPR_MAX_THREADS is None:
    NUMEXPR_MAX_THREADS = 8
    os.environ["NUMEXPR_MAX_THREADS"] = str(NUMEXPR_MAX_THREADS)

try:
    import tensorflow.compat.v1 as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)


from .submit import submit_pflow
from .resubmit import resubmit_pflow
from .info import information
from pflow import __version__


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main_parser() -> argparse.ArgumentParser:
    """pflow commandline options argument parser.
    Notes
    -----
    This function is used by documentation.
    Returns
    -------
    argparse.ArgumentParser
        the argument parser
    """
    parser = argparse.ArgumentParser(
        # description="pflow: Enhanced sampling methods under the concurrent learning framework.",
        description=information,
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n"
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    
    subparsers.add_parser(
        "ls",
        help="List all pflow tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers.add_parser(
        "dp",
        help="Something interesting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser_rm = subparsers.add_parser(
        "rm",
        help="Remove workflows of pflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_rm.add_argument(
        "WORKFLOW_ID", help="Workflow ID"
    )

    # workflow submit.
    parser_run = subparsers.add_parser(
        "submit",
        help="Submit pflow workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="\n"
    )
    parser_run.add_argument(
        "--mol", "-i", help="Initial conformation files.", dest="mol",
    )
    parser_run.add_argument(
        "--config", "-c", help="pflow configuration.", dest="config"
    )
    parser_run.add_argument(
        "--machine", "-m", help="Machine configuration.", dest="machine"
    )

    # resubmit
    parser_rerun = subparsers.add_parser(
        "resubmit",
        help="Resubmit pflow workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_rerun.add_argument(
        "WORKFLOW_ID", help="Workflow ID."
    )
    parser_rerun.add_argument(
        "--mol", "-i", help="Initial conformation files.", dest="mol",
    )
    parser_rerun.add_argument(
        "--config", "-c", help="pflow configuration.", dest="config"
    )
    parser_rerun.add_argument(
        "--machine", "-m", help="Machine configuration.", dest="machine"
    )
    parser_rerun.add_argument(
        "--iteration", "-t", help="restart from t-th iteration.", default = None, dest="iteration"
    )
    parser_rerun.add_argument(
        "--pod", "-p", help="restart from the pod.", default = None, dest="pod"
    )
    
    # Cmd
    parser_exp = subparsers.add_parser(
        "cmd",
        help="Submit pflow cmd run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_exp.add_argument(
        "--mol", "-i", help="Initial conformation files.", dest="mol",
    )
    parser_exp.add_argument(
        "--config", "-c", help="pflow configuration.", dest="config"
    )
    parser_exp.add_argument(
        "--machine", "-m", help="Machine configuration.", dest="machine"
    )
    
    # Label
    parser_label = subparsers.add_parser(
        "label",
        help="labeling MD workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_label.add_argument(
        "--mol", "-i", help="Initial conformation files.", dest="mol",
    )
    parser_label.add_argument(
        "--config", "-c", help="pflow configuration.", dest="config"
    )
    parser_label.add_argument(
        "--machine", "-m", help="Machine configuration.", dest="machine"
    )
    
    # Relabel
    parser_relabel = subparsers.add_parser(
        "relabel",
        help="Resubmit labeling MD workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_relabel.add_argument(
        "WORKFLOW_ID", help="Workflow ID."
    )
    parser_relabel.add_argument(
        "--mol", "-i", help="Initial conformation files.", dest="mol",
    )
    parser_relabel.add_argument(
        "--config", "-c", help="pflow configuration.", dest="config"
    )
    parser_relabel.add_argument(
        "--machine", "-m", help="Machine configuration.", dest="machine"
    )
    
    # train
    parser_train = subparsers.add_parser(
        "train",
        help="Train pflow neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument(
        "--data", "-d", help="Training data."
    )
    parser_train.add_argument(
        "--config", "-c", help="pflow configuration."
    )

    # NN dimension reduction.
    parser_redim = subparsers.add_parser(
        "redim",
        help="NN dimension reduction by Monte Carlo integral.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_redim.add_argument(
        "--networks", "-n", help="Training data."
    )
    parser_redim.add_argument(
        "--dim1", help="Dimension 1."
    )
    parser_redim.add_argument(
        "--dim2", help="Dimension 2."
    )

    # --version
    parser.add_argument(
        '--version', 
        action='version', 
        version='pflow v%s' % __version__,
    )

    return parser


def parse_args(args: Optional[List[str]] = None):
    """
    pflow commandline options argument parsing.
    
    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = main_parser()

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()

    return parsed_args


def parse_submit(args):
    mol_path = Path(args.mol)
    allfiles = glob.glob(str(mol_path.joinpath("*")))
    confs = []
    top_file = None
    forcefield = None
    index_file = None
    otherfiles = []
    for file in allfiles:
        if os.path.basename(file).endswith("ff"):
            forcefield = file
        elif os.path.basename(file).endswith("gro") or os.path.basename(file).endswith("lmp"):
            confs.append(file)
        elif os.path.basename(file).endswith("top"):
            top_file = file
        elif os.path.basename(file).endswith("ndx"):
            index_file = file
        else:
            otherfiles.append(file)
        
    return confs, top_file, forcefield, index_file, otherfiles


def log_ui():
    logger.info('The task is displayed on "https://127.0.0.1:2746".')
    logger.info('Artifacts (Files) are listed on "https://127.0.0.1:9001".')


def main():
    args = parse_args()
    if args.command == "submit":
        logger.info("Preparing pflow for cmd ...")
        confs, top_file, forcefield, index_file, otherfiles = parse_submit(args)
        submit_pflow(
            confs = confs,
            topology = top_file,
            pflow_config = args.config,
            machine_config = args.machine,
            forcefield = forcefield,
            index_file = index_file
        )
        log_ui()
    elif args.command == "resubmit":
        logger.info("Preparing pflow for cmd ...")
        confs, top_file, forcefield, index_file, otherfiles = parse_submit(args)
        resubmit_pflow(
            workflow_id=args.WORKFLOW_ID,
            confs = confs,
            topology = top_file,
            pflow_config = args.config,
            machine_config = args.machine,
            forcefield = forcefield,
            index_file = index_file
        )
        log_ui()
    elif args.command == "dp":
        logger.info("Molecule Simulates the Future!")
    else:
        raise RuntimeError(f"unknown command {args.command}")