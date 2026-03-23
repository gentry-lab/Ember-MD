import { ipcMain } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { IpcChannels, GenerationStats } from '../../shared/types/ipc';

function getStatsPath(): string {
  return path.join(os.homedir(), '.fraggen', 'stats.json');
}

function ensureStatsDir(): void {
  const statsDir = path.join(os.homedir(), '.fraggen');
  if (!fs.existsSync(statsDir)) {
    fs.mkdirSync(statsDir, { recursive: true });
  }
}

function getDefaultStats(): GenerationStats {
  return {
    totalMoleculesGenerated: 0,
    sessionsCount: 0,
    lastGenerationDate: null,
  };
}

export function register(): void {
  ipcMain.handle(IpcChannels.GET_STATS, async (): Promise<GenerationStats> => {
    try {
      const statsPath = getStatsPath();
      if (fs.existsSync(statsPath)) {
        const content = fs.readFileSync(statsPath, 'utf-8');
        return JSON.parse(content) as GenerationStats;
      }
      return getDefaultStats();
    } catch (error) {
      console.error('Error reading stats:', error);
      return getDefaultStats();
    }
  });

  ipcMain.handle(
    IpcChannels.UPDATE_STATS,
    async (_event, moleculeCount: number): Promise<GenerationStats> => {
      try {
        ensureStatsDir();
        const statsPath = getStatsPath();
        let stats = getDefaultStats();

        if (fs.existsSync(statsPath)) {
          try {
            const content = fs.readFileSync(statsPath, 'utf-8');
            stats = JSON.parse(content) as GenerationStats;
          } catch {
            // Use default if parse fails
          }
        }

        stats.totalMoleculesGenerated += moleculeCount;
        stats.sessionsCount += 1;
        stats.lastGenerationDate = new Date().toISOString();

        fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2), 'utf-8');
        console.log('Updated stats:', stats);

        return stats;
      } catch (error) {
        console.error('Error updating stats:', error);
        return getDefaultStats();
      }
    }
  );

  ipcMain.handle(
    IpcChannels.CHECK_JOB_EXISTS,
    async (_event, outputFolder: string, jobName: string): Promise<boolean> => {
      const jobPath = path.join(outputFolder, jobName);
      return fs.existsSync(jobPath);
    }
  );
}
