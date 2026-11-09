# Lyra Mobile

Lyra Mobile is a loopback-only HTTP service and installable PWA. A Tailscale
Serve proxy may publish it to the owner's tailnet, but the Python server itself
refuses non-loopback bind addresses.

## Start

Run from the Lyra project directory:

```powershell
python -m lyra.mobile `
  --port 8765 `
  --origin https://your-computer.your-tailnet.ts.net `
  --directory "Band calculation=C:\path\selected-on-desktop"
```

The service prints a six-digit, five-minute, one-use pairing code. Paths passed
through `--directory` are local desktop input. The browser API receives only the
generated directory ID.

Configure Tailscale Serve separately to proxy the chosen HTTPS name to
`http://127.0.0.1:8765`. Do not publish this service with router port forwarding.

## Embed

Desktop code can construct `AuthorizedDirectoryRegistry`, authorize folders
chosen with the desktop folder picker, and pass the same `ResourceRegistry` to
both `LyraTaskBackend` and `MobileService`. `LyraTaskBackend` delegates work to
the existing `SpawnWorker`, queues jobs to protect memory, and supports standard
non-projected `band`, `dos`, and `combined` previews.

Projection settings are intentionally rejected by this mobile backend version.
The `/api/capabilities` response reports that limitation to clients.
