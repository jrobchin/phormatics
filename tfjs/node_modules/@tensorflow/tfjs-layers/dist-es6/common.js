import { checkStringTypeUnionValue } from './utils/generic_utils';
var nameMap = new Map();
export var VALID_DATA_FORMAT_VALUES = ['channelsFirst', 'channelsLast'];
export function checkDataFormat(value) {
    checkStringTypeUnionValue(VALID_DATA_FORMAT_VALUES, 'DataFormat', value);
}
export var VALID_PADDING_MODE_VALUES = ['valid', 'same', 'causal'];
export function checkPaddingMode(value) {
    checkStringTypeUnionValue(VALID_PADDING_MODE_VALUES, 'PaddingMode', value);
}
export var VALID_POOL_MODE_VALUES = ['max', 'avg'];
export function checkPoolMode(value) {
    checkStringTypeUnionValue(VALID_POOL_MODE_VALUES, 'PoolMode', value);
}
var _nameScopeStack = [];
var _nameScopeDivider = '/';
export function nameScope(name, fn) {
    _nameScopeStack.push(name);
    try {
        var val = fn();
        _nameScopeStack.pop();
        return val;
    }
    catch (e) {
        _nameScopeStack.pop();
        throw e;
    }
}
function currentNameScopePrefix() {
    if (_nameScopeStack.length === 0) {
        return '';
    }
    else {
        return _nameScopeStack.join(_nameScopeDivider) + _nameScopeDivider;
    }
}
export function getScopedTensorName(tensorName) {
    if (!isValidTensorName(tensorName)) {
        throw new Error('Not a valid tensor name: \'' + tensorName + '\'');
    }
    return currentNameScopePrefix() + tensorName;
}
export function getUniqueTensorName(scopedName) {
    if (!isValidTensorName(scopedName)) {
        throw new Error('Not a valid tensor name: \'' + scopedName + '\'');
    }
    if (!nameMap.has(scopedName)) {
        nameMap.set(scopedName, 0);
    }
    var index = nameMap.get(scopedName);
    nameMap.set(scopedName, nameMap.get(scopedName) + 1);
    if (index > 0) {
        var result = scopedName + '_' + index;
        nameMap.set(result, 1);
        return result;
    }
    else {
        return scopedName;
    }
}
var tensorNameRegex = new RegExp(/^[A-Za-z][A-Za-z0-9\._\/]*$/);
export function isValidTensorName(name) {
    return name.match(tensorNameRegex) ? true : false;
}
//# sourceMappingURL=common.js.map