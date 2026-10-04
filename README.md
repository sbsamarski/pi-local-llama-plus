# pi-local-llama-plus

A personal fork of [pi-llama-cpp](https://github.com/gsanhueza/pi-llama-cpp) (0.14.0) for the
[pi coding agent](https://github.com/earendil-works/pi), tuned for a local `llama-server`
(`ik_llama.cpp`) on `127.0.0.1:9931`.

## Why the fork

The saved default model must stay **restorable even when the server is busy, loading, or temporarily
down**:

- the health probe accepts ik_llama's busy state instead of treating it as an error
- the last known model list is **cached and registered while the server is unreachable**, so
  `/model` still shows and restores your saved model
- a background retry refreshes the model list once the server answers again

The provider id is unchanged (`llama-server=<url>`), so existing settings keep working.

## Install

```bash
pi install git:github.com/sbsamarski/pi-local-llama-plus
```

or copy the folder into `~/.pi/agent/extensions/` and run `/reload`. There are no runtime npm
dependencies (`@types/node` is dev-only), so the folder is lean by design.

## Configuration

Point pi at your local server, e.g. in `~/.pi/agent/settings.json`:

```json
"providers": { "local-llama": { "baseUrl": "http://127.0.0.1:9931/v1", "apiKey": "your_llama.cpp_api_key", "apiType": "openai" } }
```

Models come from the server's `/v1/models` (registered, cached, retried in the background).

## Origin

Fork of pi-llama-cpp by Gabriel Sanhueza (MIT). All changes here target the ik_llama.cpp `llama-server`
health/busy semantics and offline model persistence.

> **Note on `tsconfig.json`:** it exists only for optional type-checking on the maintainer's machine
> (its `paths` entries point at the maintainer's global pi install). It is never used at runtime and
> does not affect loading, running, or installing this extension on another computer.
