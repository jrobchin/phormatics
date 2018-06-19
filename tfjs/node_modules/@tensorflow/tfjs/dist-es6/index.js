export * from '@tensorflow/tfjs-core';
export * from '@tensorflow/tfjs-layers';
export * from '@tensorflow/tfjs-converter';
import { version_core } from '@tensorflow/tfjs-core';
import { version_layers } from '@tensorflow/tfjs-layers';
import { version_converter } from '@tensorflow/tfjs-converter';
import { version as version_union } from './version';
export var version = {
    'tfjs-core': version_core,
    'tfjs-layers': version_layers,
    'tfjs-converter': version_converter,
    'tfjs': version_union
};
//# sourceMappingURL=index.js.map