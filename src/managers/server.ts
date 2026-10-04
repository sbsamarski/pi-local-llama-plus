import { getAgentDir, type ExtensionAPI, type ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { ApiError } from "../api/client";
import { API_TYPE, PROVIDER_NAME } from "../constants";
import { ServerStatus } from "../enums/serverStatus";
import type { SortBy } from "../interfaces/sortBy";
import { BaseModel } from "../models/baseModel";
import { Server } from "../server";
import type { LlamaSettingsManager } from "./settings";

/** Model-list comparator: negative if a sorts first, positive if b does. */
type ModelComparator = (a: BaseModel, b: BaseModel) => number;

/** One provider's entry in the last-known-model cache file. */
interface CachedProvider {
  /** When the models were captured (epoch ms). */
  savedAt: number;
  /** The registered provider model configs from the last successful scan. */
  models: ProviderModelConfig[];
}

/** Shape of the cache file on disk. */
interface ModelCache {
  providers: Record<string, CachedProvider>;
}

/** Cache file name, inside the pi agent directory. */
const CACHE_FILENAME = "llama-local-model-cache.json";

/** Delay between background re-scan attempts after a failed scan. */
const RETRY_INTERVAL_MS = 5000;

/** Background retry budget: 60 attempts x 5 s = ~5 minutes per scan cycle. */
const RETRY_MAX_ATTEMPTS = 60;

export class ServerManager {
  constructor(private readonly settings: LlamaSettingsManager) {}
  readonly failedUrls: string[] = [];
  private readonly warnings: string[] = [];
  private readonly serverList: Server[] = [];

  /** Server timeout from the last `initialize()`, reused by background retries. */
  private lastServerTimeout = 15000;

  /** Pending background retry timer, if any. */
  private retryTimer: ReturnType<typeof setTimeout> | undefined;

  /** How many background retries have run since the last successful scan. */
  private retryAttempts = 0;

  /** True while a background retry scan runs: no warnings are collected. */
  private quiet = false;

  /**
   * Live view of the server list. `update()` re-derives the list from
   * settings on every scan (in place), so `/models servers` edits apply
   * without a restart.
   */
  get servers(): readonly Server[] {
    return this.serverList;
  }

  /**
   * Verifies reachability of servers and registers the providers
   *
   * @param pi The Pi extension API
   */
  async initialize(pi: ExtensionAPI) {
    // Register the providers with the configured server timeout
    const { serverTimeout } = await this.settings.resolveTimeouts();
    this.lastServerTimeout = serverTimeout;
    await this.update(pi, serverTimeout);
  }

  /**
   * Registers one provider per server in Pi with their model configurations.
   * The manual awaiting per-server is deliberate (we want them in order)
   *
   * Servers that fail the scan still register their **last known models from
   * the cache**, so pi's saved default model stays restorable while the server
   * is down, still loading, or too slow to answer `/health`. A failed scan
   * also schedules silent background retries that refresh the registration
   * once the server responds.
   *
   * @param pi The Pi extension API
   * @param timeout (Optional) Timeout before assuming server has failed
   */
  async update(pi: ExtensionAPI, timeout?: number) {
    this.failedUrls.length = 0;

    // A manual scan (`/models`) re-arms the background retry budget.
    if (timeout === undefined) this.retryAttempts = 0;

    // Surface warnings from strict URL parsing (dropped invalid entries)
    this.warnings.push(...this.settings.takeWarnings());

    // Re-derive the server list from settings so `/models servers` edits
    // (add / remove / URL / id / name) apply on the next scan
    const fresh: Server[] = [];
    const seen = new Set<string>(); // dedupe repeated URLs (same providerId)
    for (const server of await this.settings.resolveServers()) {
      if (seen.has(server.providerId)) continue;
      seen.add(server.providerId);
      fresh.push(server);
    }

    // Unregister providers that disappeared (removed or edited away);
    // no-op for providers that were never registered
    for (const old of this.servers) {
      if (fresh.some((f) => f.providerId === old.providerId)) continue;
      pi.unregisterProvider(old.providerId);
      // Optional chain is intentional despite the non-optional type: `sse`
      // is undefined until initialize() runs (async-constructor hack — see Server)
      old.sseManager?.disconnect();
    }

    // Replace in place so the live `servers` view stays valid (D1)
    this.serverList.length = 0;
    this.serverList.push(...fresh);

    const effectiveTimeout = timeout ?? this.lastServerTimeout;
    const registrableServers = timeout
      ? await this.findRegistrableServers(effectiveTimeout)
      : this.servers;

    // Providers that ended this scan with a real (fresh or auth-pending)
    // registration; everything else falls back to the cache below.
    const registered = new Set<string>();

    // Initialization and registration
    for (const server of registrableServers) {
      try {
        await server.initialize();
        const modelConfigs = await this.registerProvider(server, pi);
        registered.add(server.providerId);
        await this.saveCache(server.providerId, modelConfigs);
      } catch (err) {
        if (err instanceof ApiError && err.type === "authentication") {
          // The server IS reachable, auth just isn't configured yet. Register
          // with the cached models when we have them (keeps the saved default
          // restorable), otherwise with an empty list so `/login` can configure
          // the key. On the next scan the provider re-initializes.
          registered.add(server.providerId);
          const cached = await this.readCache(server.providerId);
          const models = cached?.models ?? [];
          // Don't add to `failedUrls` — the health indicator should stay green.
          const message = [
            "[pi-llama-cpp-local]",
            `Server at '${server.baseUrl}' requires a valid API key.`,
            "Configure the key via `/login` or in `~/.pi/agent/auth.json`.",
          ].join("\n");
          if (!this.quiet) this.warnings.push(message);
          pi.registerProvider(server.providerId, {
            name: server.providerName,
            baseUrl: server.apiBaseUrl,
            api: API_TYPE,
            apiKey: server.getApiKey(),
            models,
          });
          continue;
        }
        this.failedUrls.push(server.baseUrl);
        continue;
      }
    }

    // Cache fallback: while a server is unreachable, register its last known
    // models so pi's saved default model is selectable and restorable. Warnings
    // are kept to the provider pi actually needs (the saved default's server).
    const defaultProvider = this.settings.resolveDefaultProvider();
    for (const server of this.servers) {
      if (registered.has(server.providerId)) continue;
      const cached = await this.readCache(server.providerId);
      if (!cached || cached.models.length === 0) {
        if (!this.quiet && server.providerId === defaultProvider) {
          this.warnings.push(
            [
              "[pi-llama-cpp-local]",
              `${PROVIDER_NAME} server at '${server.baseUrl}' did not respond, and no cached model list exists yet, so its models are not registered.`,
              "Run `/models` once the server responds to register them.",
            ].join("\n"),
          );
        }
        continue;
      }
      pi.registerProvider(server.providerId, {
        name: server.providerName,
        baseUrl: server.apiBaseUrl,
        api: API_TYPE,
        apiKey: server.getApiKey(),
        models: cached.models,
      });
      if (!this.quiet && server.providerId === defaultProvider) {
        this.warnings.push(
          [
            "[pi-llama-cpp-local]",
            `Server at '${server.baseUrl}' is not responding (down, loading, or busy), so its last known models were registered from cache (${ServerManager.formatAge(cached.savedAt)} old).`,
            "They refresh automatically as soon as the server responds.",
          ].join("\n"),
        );
      }
    }

    if (this.failedUrls.length > 0) this.scheduleRetry(pi);
    else this.cancelRetry();
  }

  /**
   * Runs concurrent health checks and returns only healthy servers.
   *
   * @param timeout Maximum time to wait for each server
   * @returns Array of servers that passed the health check
   */
  private async findRegistrableServers(timeout: number): Promise<Server[]> {
    const healthResults = await Promise.all(
      this.servers.map(async (server) => {
        const status = await server.isReady(timeout);
        return { server, status };
      }),
    );

    // Startup warnings stay focused on the server pi actually needs (the one
    // holding the saved default model). Other servers (e.g. auxiliary workers)
    // are still health-tracked and visible in `/models servers`, they just
    // don't produce warning spam.
    const defaultProvider = this.settings.resolveDefaultProvider();

    const response: Server[] = [];
    for (const { server, status } of healthResults) {
      if (status === ServerStatus.READY) {
        response.push(server);
      } else if (status === ServerStatus.TIMEOUT) {
        if (!this.quiet && server.providerId === defaultProvider) {
          const message = [
            "[pi-llama-cpp-local]",
            `${PROVIDER_NAME} server initialization for '${server.baseUrl}' took more than ${timeout} ms, so it has been skipped.`,
            "Run `/models` to retry without timeout and see all models.",
          ].join("\n");
          this.warnings.push(message);
        }
        this.failedUrls.push(server.baseUrl);
      } else {
        if (!this.quiet && server.providerId === defaultProvider) {
          const message = [
            "[pi-llama-cpp-local]",
            `${PROVIDER_NAME} server at '${server.baseUrl}' is unreachable.`,
            "Check the URL and try again. Run `/models` to retry.",
          ].join("\n");
          this.warnings.push(message);
        }
        this.failedUrls.push(server.baseUrl);
      }
    }

    return response;
  }

  /**
   * Creates a Pi provider for the given server.
   *
   * @param server The server
   * @param pi The Pi API
   * @returns The registered model configs (also written to the cache)
   */
  private async registerProvider(
    server: Server,
    pi: ExtensionAPI,
  ): Promise<ProviderModelConfig[]> {
    const { apiBaseUrl, models, providerId, providerName } = server;
    const apiKey = server.getApiKey();
    const modelConfigs = await Promise.all(
      models.map((m) => m.toProviderConfig()),
    );

    pi.registerProvider(providerId, {
      name: providerName,
      baseUrl: apiBaseUrl,
      api: API_TYPE,
      apiKey: apiKey,
      models: modelConfigs,
    });

    return modelConfigs;
  }

  /**
   * Schedules one silent background re-scan, unless the retry budget is spent
   * or a retry is already pending. The chain continues from `update()` until
   * the scan succeeds or the budget runs out (~5 minutes). After a session
   * replacement the old context's registrations are dropped by pi, so stale
   * chains are harmless; each new session starts its own chain.
   *
   * @param pi The Pi extension API
   */
  private scheduleRetry(pi: ExtensionAPI) {
    if (this.retryTimer) return;
    if (this.retryAttempts >= RETRY_MAX_ATTEMPTS) return;
    this.retryAttempts += 1;
    this.retryTimer = setTimeout(() => {
      this.retryTimer = undefined;
      this.quiet = true;
      void this.update(pi, this.lastServerTimeout)
        .catch(() => {})
        .finally(() => {
          this.quiet = false;
        });
    }, RETRY_INTERVAL_MS);
    // Never keep the process alive just for a retry.
    this.retryTimer.unref?.();
  }

  /** Cancels the pending background retry and resets its budget. */
  private cancelRetry() {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = undefined;
    }
    this.retryAttempts = 0;
  }

  // ---------------------------------------------------------------------------
  // Last-known-model cache
  // ---------------------------------------------------------------------------

  /** Path of the cache file inside the pi agent directory. */
  private cachePath(): string {
    return join(getAgentDir(), CACHE_FILENAME);
  }

  /**
   * Reads one provider's cached model list.
   *
   * @param providerId The provider id
   * @returns The cached entry, or `undefined` when absent or unreadable
   */
  private async readCache(
    providerId: string,
  ): Promise<CachedProvider | undefined> {
    try {
      const raw = await readFile(this.cachePath(), "utf8");
      const cache = JSON.parse(raw) as ModelCache;
      return cache.providers?.[providerId];
    } catch {
      return undefined;
    }
  }

  /**
   * Persists one provider's model configs after a successful scan. Never
   * caches an empty list (an auth-pending registration must not shadow a
   * previously cached working list).
   *
   * Best effort: a failed write only costs cache freshness.
   *
   * @param providerId The provider id
   * @param models The freshly registered model configs
   */
  private async saveCache(
    providerId: string,
    models: ProviderModelConfig[],
  ): Promise<void> {
    if (models.length === 0) return;
    let cache: ModelCache = { providers: {} };
    try {
      cache = JSON.parse(await readFile(this.cachePath(), "utf8")) as ModelCache;
      cache.providers ??= {};
    } catch {
      cache = { providers: {} };
    }
    cache.providers[providerId] = { savedAt: Date.now(), models };
    try {
      await writeFile(this.cachePath(), JSON.stringify(cache, null, 2), "utf8");
    } catch {
      // Best effort — a stale cache is still better than none.
    }
  }

  /** Formats a saved-at timestamp as a short human-readable age. */
  private static formatAge(savedAt: number): string {
    const seconds = Math.max(1, Math.round((Date.now() - savedAt) / 1000));
    if (seconds < 90) return `${seconds}s`;
    if (seconds < 5400) return `${Math.round(seconds / 60)}min`;
    if (seconds < 172800) return `${Math.round(seconds / 3600)}h`;
    return `${Math.round(seconds / 86400)}d`;
  }

  /**
   * Returns warnings collected during initialization.
   */
  getWarnings(): string[] {
    const warnings = [...this.warnings];
    this.warnings.length = 0;

    return warnings;
  }

  /**
   * Returns the server for a given model.
   *
   * @param model - The model to find the server for
   * @returns The server containing the model, or `undefined` when no
   * current server matches (e.g. removed while a model was loading)
   */
  getServer(model: BaseModel): Server | undefined {
    return this.servers.find((s) => s.baseUrl === model.serverUrl);
  }

  /**
   * Returns all models from all servers, sorted by the configured sort mode.
   * Servers maintain their order from `llamaSettings`; sorting only applies
   * to models within each server.
   *
   * @returns Flat array of all models across all servers
   */
  async getAllModels(): Promise<BaseModel[]> {
    const sortBy = await this.settings.resolveSortBy();

    if (sortBy === "api") {
      return this.servers.flatMap((s) => s.models);
    }

    const sorter = ServerManager.SORTERS[sortBy];
    return this.servers.flatMap((s) => [...s.models].sort(sorter));
  }

  private static sortByIdAsc(a: BaseModel, b: BaseModel): number {
    return a.id.localeCompare(b.id);
  }

  private static sortByIdDesc(a: BaseModel, b: BaseModel): number {
    return b.id.localeCompare(a.id);
  }

  /** Name ascending, with ID as tiebreaker. */
  private static sortByNameAsc(a: BaseModel, b: BaseModel): number {
    const cmp = a.name.localeCompare(b.name);
    return cmp !== 0 ? cmp : a.id.localeCompare(b.id);
  }

  /** Name descending, with ID as tiebreaker. */
  private static sortByNameDesc(a: BaseModel, b: BaseModel): number {
    const cmp = b.name.localeCompare(a.name);
    return cmp !== 0 ? cmp : a.id.localeCompare(b.id);
  }

  /**
   * Comparators for sorting models within each server.
   */
  private static readonly SORTERS: Record<
    Exclude<SortBy, "api">,
    ModelComparator
  > = {
    asc: ServerManager.sortByIdAsc,
    desc: ServerManager.sortByIdDesc,
    "asc-name": ServerManager.sortByNameAsc,
    "desc-name": ServerManager.sortByNameDesc,
  };
}
