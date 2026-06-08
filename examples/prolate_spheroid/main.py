# DETERMINISTIC (CLAUDE.md Sec. 2.9)
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import train
import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store run artifacts.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    elif FLAGS.config.mode == "eval":
        evaluate.evaluate(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Unknown mode: {FLAGS.config.mode}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
