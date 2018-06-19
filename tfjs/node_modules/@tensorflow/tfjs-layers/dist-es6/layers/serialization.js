import { serialization } from '@tensorflow/tfjs-core';
import { deserializeKerasObject } from '../utils/generic_utils';
export function deserialize(config, customObjects) {
    if (customObjects === void 0) { customObjects = {}; }
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'layer');
}
//# sourceMappingURL=serialization.js.map