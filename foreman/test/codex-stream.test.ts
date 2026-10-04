import { describe, expect, it, vi } from 'vitest';
import type { Foreman } from '../src/foreman.js';
import { CodexStreamMapper } from '../src/agents/codex/stream.js';
import { tailLines } from '../src/util/text.js';

function fixture() {
  const agentLog = vi.fn();
  const mapper = new CodexStreamMapper({ agentLog, setAgent: vi.fn() } as unknown as Foreman, 'kit', '/project', 'worker');
  const retained = () => (mapper as unknown as { commandOutput: Map<string, string> }).commandOutput;
  return { mapper, agentLog, retained };
}

describe('Codex streamed command output', () => {
  it('bounds retained output from a noisy command while delivering every live delta', () => {
    const { mapper, agentLog, retained } = fixture();
    const chunk = 'build output line\n'.repeat(4096);
    for (let i = 0; i < 256; i++) mapper.handle('item/commandExecution/outputDelta', { itemId: 'build', delta: chunk });
    expect(retained().get('build')!.length).toBeLessThanOrEqual(16_384);
    expect(agentLog).toHaveBeenCalledTimes(256);
    mapper.handle('item/commandExecution/outputDelta', { itemId: 'build', delta: 'fatal: compilation failed\n' });
    mapper.handle('item/completed', { item: { id: 'build', type: 'commandExecution', exitCode: 1 } });
    expect(agentLog.mock.lastCall?.[2]).toBe(tailLines(chunk + 'fatal: compilation failed\n', 8, 900));
    expect(retained().size).toBe(0);
  });

  it('keeps independent command tails across fragmented Unicode and CRLF output', () => {
    const { mapper, agentLog } = fixture();
    mapper.handle('item/commandExecution/outputDelta', { itemId: 'a', delta: 'first\r' });
    mapper.handle('command/exec/outputDelta', { itemId: 'b', delta: 'other command\n' });
    mapper.handle('item/commandExecution/outputDelta', { itemId: 'a', delta: '\nlast 🚀\n' });
    mapper.handle('item/completed', { item: { id: 'a', type: 'commandExecution', exitCode: 0 } });
    expect(agentLog.mock.lastCall?.[2]).toBe('first\nlast 🚀');
    mapper.handle('item/completed', { item: { id: 'b', type: 'commandExecution', exitCode: 0 } });
    expect(agentLog.mock.lastCall?.[2]).toBe('other command');
  });

  it('uses authoritative aggregate output and releases orphaned buffers at turn completion', () => {
    const { mapper, agentLog, retained } = fixture();
    mapper.handle('item/commandExecution/outputDelta', { itemId: 'a', delta: 'partial' });
    mapper.handle('item/completed', { item: { id: 'a', type: 'commandExecution', aggregatedOutput: 'final output', exitCode: 0 } });
    expect(agentLog.mock.lastCall?.[2]).toBe('final output');
    mapper.handle('item/commandExecution/outputDelta', { itemId: 'orphan', delta: 'unfinished output' });
    mapper.handle('turn/completed', { turn: { status: 'interrupted' } });
    expect(retained().size).toBe(0);
  });
});
