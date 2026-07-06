---
name: vastai-experiment
description: Use for connecting to and destroying Vast.ai GPU instances. Trigger when the user mentions Vast.ai, a GPU instance, or wants to run experiments on a remote GPU.
---

# Vast.ai Experiment Helper

Use this skill when the user wants to connect to an existing Vast.ai GPU instance, verify GPU availability, run a quick remote command, or destroy the instance after an experiment is finished.

## Scope

- Do connect to an existing instance.
- Do verify the GPU is available after connecting.
- Do destroy the instance only when the user explicitly says the experiment is done.
- Do not search for or create new instances.
- Assume only one active instance is being managed at a time.

## Connect To An Instance

1. List current instances:

```bash
vastai show instances
```

2. Get the SSH URL for the target instance:

```bash
vastai ssh-url <instance_id>
```

3. Connect over SSH:

```bash
ssh root@<host> -p <port>
```

4. If authentication fails, retry with an explicit key:

```bash
ssh -i ~/.ssh/id_rsa root@<host> -p <port>
```

5. If needed, inspect available registered SSH keys:

```bash
vastai show ssh-keys
```

6. After connecting, verify the GPU is visible:

```bash
nvidia-smi
```

## Configure Git And GitHub Access

Run this immediately after connecting to the Vast.ai instance if the user needs to access GitHub from the remote machine.

1. Configure the Git identity:

```bash
git config --global user.name "YiconRain"
git config --global user.email "fuckirving@sjtu.edu.cn"
```

2. Generate a new SSH key for GitHub access:

```bash
ssh-keygen -t ed25519 -C "fuckirving@sjtu.edu.cn"
```

3. Show the public key so it can be added to the GitHub account:

```bash
cat ~/.ssh/id_ed25519.pub
```

4. Add the displayed public key to the user's GitHub account.

5. Optionally test SSH access to GitHub:

```bash
ssh -T git@github.com
```

## Destroy An Instance

Warning: destroying an instance is permanent and all data on that instance will be lost.

1. Destroy with confirmation skipped:

```bash
vastai destroy instance <instance_id> -y
```

2. Or destroy with interactive confirmation:

```bash
vastai destroy instance <instance_id>
```

3. Verify the instance is gone:

```bash
vastai show instances
```

## Useful Commands

| Action | Command |
| --- | --- |
| List instances | `vastai show instances` |
| Get SSH URL | `vastai ssh-url <id>` |
| Run remote command | `vastai execute <id> "command"` |
| Show SSH keys | `vastai show ssh-keys` |
| Destroy instance | `vastai destroy instance <id> -y` |

## Workflow

### User asks to connect

1. Run `vastai show instances`.
2. Choose the active instance ID.
3. Run `vastai ssh-url <instance_id>`.
4. Connect with `ssh`.
5. Run `nvidia-smi` after login.
6. If the user needs GitHub access, configure Git identity and generate an SSH key immediately.
7. Add the generated public key to GitHub, then verify with `ssh -T git@github.com`.

### User asks to destroy

1. Warn that data loss is permanent.
2. Run `vastai destroy instance <instance_id> -y`.
3. Confirm removal with `vastai show instances`.

## Notes

- The user manually starts instances.
- Do not create or search for new machines.
- If the user needs repository access, set up Git and GitHub SSH right after connecting.
- Only destroy when the user explicitly confirms the experiment is complete.

