import logging
import os
import tempfile
import uuid
from typing import List

import tensorflow as tf
from tensorflow.python.framework.errors_impl import AlreadyExistsError

logger = logging.getLogger(__name__)


def gfile_copy_tree(src, dest, overwrite=False):
    """
    Recursively copy src directory to dest
    :param src: source directory path
    :param dest: destination path
    :param overwrite: allow overwriting
    """
    for dir, subdirs, files in tf.io.gfile.walk(src):
        rel_path = os.path.relpath(dir, src)
        dest_dir = os.path.join(dest, rel_path) if rel_path != "." else dest
        print(f"Making directory \"{dest_dir}\"")
        tf.io.gfile.makedirs(dest_dir)
        for f in files:
            src_path = os.path.join(dir, f)
            dest_path = os.path.join(dest_dir, f)
            print(f"Copying file \"{src_path}\" to \"{dest_path}\"")
            try:
                tf.io.gfile.copy(src_path, dest_path, overwrite=overwrite)
            except AlreadyExistsError:
                print(f"Not overwriting file {src_path}.")


class RemoteIOArgsContext(object):
    """
    A context manager class within which command line args that contains remote paths will be mapped to local
    paths at entry of context, and mapped back to remote path at the exit
    """
    def __init__(self,
                 args=None,
                 input_args: List[str] = [],
                 output_args: List[str] = [],
                 enable_upload=False):
        """
        :param args: an object that stores argument values, e.g. Namespace; object
        :param input_args: Name of input remote path arguments, files in which will be downloaded; list of str
        :param output_args: Name of output remote path arguments, files generated in which will be uploaded; list of str
        :param enable_upload: Enable uploading files, should set to False if rank is non-zero; bool
        """
        self.__args = args
        self.__input_args = set(input_args)
        self.__output_args = set(output_args)
        self.__remote_paths = {}
        self.__tmp_path = None 
        self.enable_upload = enable_upload

    def __enter__(self):
        self.__tmp_path = tempfile.TemporaryDirectory()
        for arg in self.__input_args:
            remote_path = getattr(self.__args, arg)
            self.__remote_paths[arg] = remote_path
            if tf.io.gfile.exists(remote_path):
                if tf.io.gfile.isdir(remote_path):
                    local_path = os.path.join(self.__tmp_path.name, f"{uuid.uuid4()}")
                    logger.info(f"Mapping \"{remote_path}\" to \"{local_path}\"", flush=True)
                    gfile_copy_tree(remote_path, local_path)
                else:
                    local_path = os.path.join(self.__tmp_path.name, f"{uuid.uuid4()}/{os.path.basename(remote_path)}")
                    logger.info(f"Mapping \"{remote_path}\" to \"{local_path}\"", flush=True)
                    tf.io.gfile.makedirs(os.path.dirname(local_path))
                    tf.io.gfile.copy(remote_path, local_path)
                setattr(self.__args, arg, local_path)
            else:
                raise FileNotFoundError(f"File not found: {remote_path}")

        for arg in self.__output_args:
            # If the same arg has been mapped as an input, then no need to do it again
            if arg not in self.__input_args:
                remote_path = getattr(self.__args, arg)
                local_path = os.path.join(self.__tmp_path.name, f"{uuid.uuid4()}")
                print(f"Mapping \"{remote_path}\" to \"{local_path}\"", flush=True)
                os.makedirs(local_path, exist_ok=True)
                setattr(self.__args, arg, local_path)
                self.__remote_paths[arg] = remote_path
        return self.__args

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_upload:
            for arg in self.__output_args:
                remote_path = self.__remote_paths[arg]
                local_path = getattr(self.__args, arg)
                print(f"Uploading \"{local_path}\" to \"{remote_path}\"", flush=True)
                gfile_copy_tree(local_path, remote_path)
                setattr(self.__args, arg, remote_path)

        self.__tmp_path.cleanup()

