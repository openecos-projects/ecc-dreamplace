"""Detect system pkg-config search paths at repo phase."""

def _pkg_config_path_repo_impl(repository_ctx):
    result = repository_ctx.execute(["pkg-config", "--variable", "pc_path", "pkg-config"])
    if result.return_code != 0:
        fail(
            "pkg-config failed (return code {}): {}\n{}".format(
                result.return_code,
                result.stdout.strip(),
                result.stderr.strip(),
            ),
        )
    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file("defs.bzl", 'PKG_CONFIG_PATH = "%s"' % result.stdout.strip())

_pkg_config_path_repo = repository_rule(
    implementation = _pkg_config_path_repo_impl,
    local = True,
)

def _pkg_config_ext_impl(module_ctx):
    _pkg_config_path_repo(name = "pkg_config_path")

pkg_config = module_extension(
    implementation = _pkg_config_ext_impl,
)
