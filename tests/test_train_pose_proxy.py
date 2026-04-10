from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_pose


class TrainPoseProxyTests(unittest.TestCase):
    def test_apply_wandb_proxy_env_sets_expected_variables(self) -> None:
        config = {
            "wandb": {
                "enabled": True,
                "proxy": {
                    "enabled": True,
                    "url": "http://127.0.0.1:10080",
                    "no_proxy": ["127.0.0.1", "localhost"],
                },
            }
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            train_pose.apply_wandb_proxy_env(config)
            self.assertEqual(os.environ["HTTP_PROXY"], "http://127.0.0.1:10080")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:10080")
            self.assertEqual(os.environ["http_proxy"], "http://127.0.0.1:10080")
            self.assertEqual(os.environ["https_proxy"], "http://127.0.0.1:10080")
            self.assertEqual(os.environ["NO_PROXY"], "127.0.0.1,localhost")
            self.assertEqual(os.environ["no_proxy"], "127.0.0.1,localhost")

    def test_apply_wandb_proxy_env_is_noop_when_disabled(self) -> None:
        config = {
            "wandb": {
                "enabled": True,
                "proxy": {
                    "enabled": False,
                    "url": "http://127.0.0.1:10080",
                    "no_proxy": ["127.0.0.1", "localhost"],
                },
            }
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            train_pose.apply_wandb_proxy_env(config)
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertNotIn("NO_PROXY", os.environ)

    def test_normalize_wandb_proxy_config_accepts_string_no_proxy(self) -> None:
        config = {
            "wandb": {
                "enabled": True,
                "proxy": {
                    "enabled": True,
                    "url": "http://127.0.0.1:10080",
                    "no_proxy": "localhost",
                },
            }
        }
        train_pose.normalize_wandb_proxy_config(config)
        self.assertEqual(config["wandb"]["proxy"]["no_proxy"], ["localhost"])


if __name__ == "__main__":
    unittest.main()
