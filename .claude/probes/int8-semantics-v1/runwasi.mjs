import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
const wasi = new WASI({ version: 'preview1', args: [], env: {} });
const mod = await WebAssembly.compile(await readFile(process.argv[2]));
const inst = await WebAssembly.instantiate(mod, wasi.getImportObject());
wasi.start(inst);
