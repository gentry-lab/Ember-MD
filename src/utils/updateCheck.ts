// Copyright (c) 2026 Ember Contributors. MIT License.

const REPO = 'gentry-lab/Ember-MD';

export interface UpdateInfo {
  version: string;
  url: string;
}

function parseVersion(tag: string): number[] {
  return tag.replace(/^v/, '').split('.').map(Number);
}

function isNewer(remote: number[], local: number[]): boolean {
  for (let i = 0; i < 3; i++) {
    const r = remote[i] || 0;
    const l = local[i] || 0;
    if (r > l) return true;
    if (r < l) return false;
  }
  return false;
}

export async function checkForUpdate(): Promise<UpdateInfo | null> {
  try {
    const appVersion = await window.electronAPI.getAppVersion();
    const resp = await fetch(
      `https://api.github.com/repos/${REPO}/releases/latest`,
      { headers: { 'Accept': 'application/vnd.github.v3+json' }, signal: AbortSignal.timeout(10000) }
    );
    if (!resp.ok) return null;
    const data = await resp.json();
    const remoteVersion = parseVersion(data.tag_name);
    const localVersion = parseVersion(appVersion);
    if (!isNewer(remoteVersion, localVersion)) return null;
    return { version: data.tag_name, url: data.html_url };
  } catch {
    return null;
  }
}
