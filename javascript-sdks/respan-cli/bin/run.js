#!/usr/bin/env node
import {execute} from '@oclif/core';

const args = process.argv.slice(2);
if (args.length === 0 || (args.length === 1 && args[0] === '--help')) {
  const {printBanner} = await import('../dist/lib/banner.js');
  printBanner();
}

await execute({dir: import.meta.url});
