# Playground 

Playground tasks for newcomers!

## 1. Environments

Fork this repository to your own github account:

![image](./docs/imgs/fork.png)

Start a docker from image `playground:latest`:

```bash
# Params:
#   <container-name>: Your container name. Can be anything.
#   <project-home>: A directory that stores all your projects. This directory should be shared by all containers.
#   <proxy-addr>: Your proxy address.
#   <container-port>: A port that nobody has used. For example: 1145.
docker run --gpus all --name <container-name>  \
    -v $HOME/<project-home>:/root/<project-home>  \
    -e HTTP_PROXY=<proxy-addr>  \
    -e HTTPS_PROXY=<proxy-addr>  \
    -e http_proxy=<proxy-addr>  \
    -e https_proxy=<proxy-addr>  \
    -it -p <container-port>:22 --entrypoint /bin/bash playground:latest
```

After you are inside the docker, clone this repo:

```bash
cd /root/<project-home>
git clone https://github.com/<your-github-account-name>/playground.git
```

## 2. Tasks

| No. | Discription | Directory | Document |
|:---:|:---:|:---:|:---:|
| 1 | CUDA Programming: GEMM | [./task-1](./task-1) | [READEME.md](./task-1/README.md) |

