{
  description = "Flake for setting up the environment for homl exercises.";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];

      perSystem = {pkgs, ...}: {
        devShells.default =
          (pkgs.buildFHSUserEnv {
            name = "homl-micromamba";

            targetPkgs = ps:
              with ps; [
                micromamba
                graphviz-nox
                protobuf
              ];

            profile = ''
              set -e
              eval "$(micromamba shell hook --shell=posix)"
              export MAMBA_ROOT_PREFIX="$(realpath './.micromamba')"
              export CONDA_OVERRIDE_CUDA="11.8"

              env="homl"
              if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$env" ]; then
                micromamba create -qn "$env"
                test -f environment.yaml \
                  && micromamba install -yn "$env" -f environment.yaml \
                  || micromamba install -yn "$env" -c conda-forge python=3.10
                test -f requirements.txt \
                  && micromamba run -n "$env" pip install -r requirements.txt \
                  || :
              fi

              export XLA_FLAGS="--xla_gpu_cuda_data_dir=$MAMBA_ROOT_PREFIX/envs/$env"
              micromamba activate "$env"
              set +e
            '';
          })
          .env;
      };
    };
}
