# @respan/instrumentation-beeai

Respan instrumentation plugin for BeeAI Framework.

This package wraps `@arizeai/openinference-instrumentation-beeai` and routes
BeeAI OpenInference spans through the Respan OpenInference translator.

## Install

```bash
npm install @respan/respan @respan/instrumentation-beeai \
  @arizeai/openinference-instrumentation-beeai beeai-framework@0.1.13 @ai-sdk/openai@^1.3.13 zod@^3.24.2
```

## Supported Versions

This package delegates to `@arizeai/openinference-instrumentation-beeai`, which instruments `beeai-framework >=0.1.9 <0.1.14`. Use `beeai-framework@0.1.13` for the current compatible examples.

## Usage

```typescript
import * as beeaiFramework from "beeai-framework";
import { BeeAIInstrumentation as OpenInferenceBeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import { Respan } from "@respan/respan";
import { BeeAIInstrumentor } from "@respan/instrumentation-beeai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new BeeAIInstrumentor({
      sdkModule: beeaiFramework,
      instrumentationClass: OpenInferenceBeeAIInstrumentation,
    }),
  ],
});
await respan.initialize();
```

Pass `sdkModule` when running BeeAI through ESM so the underlying OpenInference
instrumentor can patch the module instance used by your app. Pass the
application-resolved `instrumentationClass` alongside it when using linked
packages, pnpm, or a bundler. This keeps the OpenInference instrumentor and the
BeeAI SDK in the same module realm, which its `instanceof ChatModel` serializer
checks require for model, input/output, and token attributes.
