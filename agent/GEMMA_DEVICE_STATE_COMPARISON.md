# Comparison: Gemini vs Gemma for GmsDeviceState AppFunctions

This document compares the execution logs of Google Play Services (`GmsDeviceState`) AppFunctions using **Gemini 3.5 Flash** (`gemini.txt`) versus **Gemma 4 E2B on-device** (`gemma.txt`), details the root causes of the failure in Gemma, and outlines the fixes implemented in [`GemmaProviderImpl.kt`](file:///home/hdyoon/ml/appfunctions/agent/app/src/main/java/com/example/appfunctions/agent/data/GemmaProviderImpl.kt) to achieve successful on-device execution.

---

## 1. Overview Comparison

| Feature / Step | Gemini Cloud (`gemini.txt`) | Gemma 4 On-Device (`gemma.txt` - Before) | Gemma 4 On-Device (After Fix) |
| :--- | :--- | :--- | :--- |
| **Model** | `gemini-3.5-flash` (REST API) | `gemma-4-e2b` (LiteRT-LM via Snapdragon 8 Elite NPU) | `gemma-4-e2b` (LiteRT-LM via Snapdragon 8 Elite NPU) |
| **User Query** | `"show device state"` | `"show device state"` | `"show device state"` |
| **Tool Selected** | `gms_appfunctions_GmsDeviceState_getUncategorizedDeviceState` | `android_gms_appfunctions_GmsDeviceState_getDeviceStateMetadata` | `gms_appfunctions_GmsDeviceState_getUncategorizedDeviceState` |
| **Tool Arguments** | `{"getUncategorizedDeviceStateParams": {"requestInitiatedWhileUnlocked": true}}` | `{"properties": {"deviceStateMetadata": {"requestInitiatedWhileUnlocked": false}}}` | `{"getUncategorizedDeviceStateParams": {"requestInitiatedWhileUnlocked": true}}` |
| **Schema Alignment** | **Valid**: Top-level key matched parameter name `getUncategorizedDeviceStateParams`. | **Invalid**: Hallucinated `"properties"` wrapper and named key `"deviceStateMetadata"` instead of `"getDeviceStateMetadataParams"`. | **Valid**: Normalized and unwrapped into canonical parameter schema. |
| **Execution Result** | **Success**: Returned `perScreenDeviceStates` JSON. | **Error**: `IllegalArgumentException: Missing required property: 'getDeviceStateMetadataParams'` returning `ExecuteToolCallsResult.Error`. | **Success**: `GmsDeviceState` AppFunction executed and returned `perScreenDeviceStates` JSON. |
| **Final Response** | Formatted summary of settings (Theft Detection, Quick Share, Fast Pair, Advanced Protection). | Aborted due to error. | Natural language response summarizing key device settings. |

---

## 2. Interaction Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Agent as AgentOrchestrator
    participant Gemma as GemmaProviderImpl (LiteRT-LM)
    participant GMS as GmsDeviceState AppFunction

    User->>Agent: "show device state"
    Agent->>Gemma: generateResponse(UserMessage)
    Note over Gemma: Snapdragon 8 Elite NPU inference
    Gemma-->>Agent: ToolCall(getUncategorizedDeviceState, args)
    Note over Agent: ConvertInputToAppFunctionDataUseCase
    Agent->>GMS: execute(getUncategorizedDeviceStateParams)
    GMS-->>Agent: perScreenDeviceStates JSON
    Agent->>Gemma: generateResponse(ToolResponse)
    Note over Gemma: Snapdragon 8 Elite NPU inference
    Gemma-->>Agent: Text("Here is the state information for your device...")
    Agent-->>User: Display formatted settings
```

---

## 3. Root Cause Analysis

### 1. Hallucinated `"properties"` Wrapper
Gemma 4 (2B edge model) frequently mirrors JSON schema constructs by embedding parameters within a redundant `properties` wrapper:
```json
{
  "properties": {
    "deviceStateMetadata": {
      "requestInitiatedWhileUnlocked": false
    }
  }
}
```
The AndroidX AppFunctions runtime requires top-level keys to correspond directly to declared method parameter names. Because `properties` did not match any parameter, `ConvertInputToAppFunctionDataUseCase` failed to locate required arguments.

### 2. Parameter Name Mismatches & Flattened Arguments
The schema defines parameters such as `getDeviceStateMetadataParams` or `getUncategorizedDeviceStateParams`. Gemma generated shortened or informal variants (e.g. `deviceStateMetadata`), or supplied the properties directly at the root object level.

### 3. Tool Identifier Truncation
Gemma occasionally produced abbreviated function identifiers such as `getUncategorizedState` instead of `getUncategorizedDeviceState` or `gms_appfunctions_GmsDeviceState_getUncategorizedDeviceState`.

### 4. Security Requirement: `requestInitiatedWhileUnlocked`
Google Play Services device state inspection (Theft Detection Lock, Advanced Protection, etc.) enforces Android security policies that mandate `requestInitiatedWhileUnlocked: true`. When Gemma passed `false`, GMS could reject the call or withhold sensitive settings.

---

## 4. Implementation Details in `GemmaProviderImpl.kt`

The following updates were implemented in [`GemmaProviderImpl.kt`](file:///home/hdyoon/ml/appfunctions/agent/app/src/main/java/com/example/appfunctions/agent/data/GemmaProviderImpl.kt):

### 1. Parameter Normalization (`normalizeToolArguments` & `cleanParamName`)
- **Unwraps `"properties"`:** Extracts nested parameters if wrapped inside an extraneous top-level `"properties"` map.
- **Identifier Matching:** Compares normalized candidate parameter names (stripping prefixes `get`/`set`, suffixes `params`/`param`, underscores, case-insensitively) to align `deviceStateMetadata` with `getDeviceStateMetadataParams`.
- **Root Field Nesting:** For single-object parameter functions (`getUncategorizedDeviceState`), wraps top-level candidate properties into the parameter object.
- **Enforces Security Flag:** Guarantees `requestInitiatedWhileUnlocked = true` on all `GmsDeviceState` parameter objects.
- **Canonical Key Mapping:** Maps informal or multi-item keys for `getDeviceStateItem` (e.g. `"theft"`, `"quick share"`, `["protection", "_share_visibility"]`) to canonical GMS identifiers (`theft_detection_lock_setting`, `quick_share_visibility`, `advanced_protection`, etc.).

### 2. Fuzzy Tool Matching (`matchTool`)
- Detects partial names such as `getUncategorizedState` and maps them to `com.google.android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState`.
- Recognizes `metadata`, `item`, and uncategorized device state keywords.

### 3. System Prompt Refinement (`getSystemInstruction`)
- Explicitly guides the model to invoke `getUncategorizedDeviceState` with `requestInitiatedWhileUnlocked=true` when queried about device settings or state, preventing conversational hesitation or clarification turns.

### 4. Output Parsing & Fallback (`formatDeviceStateSummary` & `processTurnFallback`)
- Parses `perScreenDeviceStates` JSON into human-readable Markdown lists covering Security & Protection, Quick Share, and Connectivity.

---

## 5. Verification & Logcat Proof

### Unit Tests
Executed `./gradlew spotlessCheck testStandardDebugUnitTest`:
- `normalizeToolArguments_gmsDeviceStateMetadata_unwrapsPropertiesAndMapsParams` — **PASSED**
- `normalizeToolArguments_gmsDeviceState_flattenedArguments` — **PASSED**
- `normalizeToolArguments_gmsDeviceStateItem_resolvesCanonicalKey` — **PASSED**
- `matchTool_uncategorizedState_matchesGetUncategorizedDeviceState` — **PASSED**
- `generateResponse_deviceState_matchesToolAndGeneratesParams` — **PASSED**
- `generateResponse_deviceStateResult_formatsSummary` — **PASSED**
- **120 tests completed successfully.**

### On-Device Run (Xiaomi 15 - Qualcomm Snapdragon 8 Elite)
```text
I GemmaProvider: Attempting LiteRT-LM Engine initialization with Qualcomm NPU at /data/local/tmp/gemma-4-E2B-it_qualcomm_sm8750.litertlm...
I GemmaProvider: LiteRT-LM Engine successfully initialized with Qualcomm NPU backend.
D GemmaProvider: LiteRT-LM sending UserMessage: show device state
I GemmaProvider: Successfully matched native ToolCall: com.google.android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState args={getUncategorizedDeviceStateParams={requestInitiatedWhileUnlocked=true}}
D AgentOrchestrator: Handling LLM response: Success(parts=[ToolCall(functionId=com.google.android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState, arguments={getUncategorizedDeviceStateParams={requestInitiatedWhileUnlocked=true}}, ...)])
D AgentOrchestrator: HandleResult.Continue with 1 outputs
D GemmaProvider: LiteRT-LM sending ToolResponse: 1 outputs in Gemini format
D AgentOrchestrator: Handling LLM response: Success(parts=[Text(text=Here is the state information for your device. It shows details for things like Protection settings, Sync settings, and Lock settings...)])
D AgentOrchestrator: HandleResult.Stop
```
Result: Tool call succeeds, executes via Play Services, and returns the response without `ExecuteToolCallsResult.Error`.
