/*
 * Copyright 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.appfunctions.agent.data

import android.util.Log
import androidx.appfunctions.metadata.AppFunctionBooleanTypeMetadata
import androidx.appfunctions.metadata.AppFunctionComponentsMetadata
import androidx.appfunctions.metadata.AppFunctionMetadata
import androidx.appfunctions.metadata.AppFunctionObjectTypeMetadata
import androidx.appfunctions.metadata.AppFunctionParameterMetadata
import androidx.appfunctions.metadata.AppFunctionResponseMetadata
import androidx.appfunctions.metadata.AppFunctionStringTypeMetadata
import com.example.appfunctions.agent.domain.LlmInput
import com.example.appfunctions.agent.domain.LlmResponse
import com.example.appfunctions.agent.domain.LlmResponsePart
import com.example.appfunctions.agent.domain.ToolOutput
import io.mockk.every
import io.mockk.mockkStatic
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class Gemma4E2bProviderImplTest {
    private val toolConverter = GeminiToolConverter()
    private lateinit var provider: Gemma4E2bProviderImpl

    @Before
    fun setUp() {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.i(any(), any()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        provider = Gemma4E2bProviderImpl(toolConverter)
    }

    @Test
    fun generateResponse_userMessage_returnsGemmaResponse() =
        runBlocking {
            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("Hello Gemma 4"),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                )

            assertTrue(response is LlmResponse.Success)
            val success = response as LlmResponse.Success
            assertNotNull(success.interactionId)
            val textParts = success.parts.filterIsInstance<LlmResponsePart.Text>()
            assertTrue(textParts.isNotEmpty())
            assertTrue(textParts.any { it.text.contains("Gemma 4") })
        }

    @Test
    fun generateResponse_e2bModel_returnsGemmaResponse() =
        runBlocking {
            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("Hello Gemma 4 E2B"),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e2b",
                )

            assertTrue(response is LlmResponse.Success)
            val success = response as LlmResponse.Success
            assertNotNull(success.interactionId)
            val textParts = success.parts.filterIsInstance<LlmResponsePart.Text>()
            assertTrue(textParts.isNotEmpty())
            assertTrue(textParts.any { it.text.contains("gemma-4-e2b") })
        }

    @Test
    fun generateResponse_multiTurn_maintainsSession() =
        runBlocking {
            val firstResponse =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("First query to Gemma"),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                ) as LlmResponse.Success

            val secondResponse =
                provider.generateResponse(
                    previousInteractionId = firstResponse.interactionId,
                    input = LlmInput.UserMessage("Follow up query"),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                )

            assertTrue(secondResponse is LlmResponse.Success)
        }

    @Test
    fun generateResponse_toolExecution_handlesToolCalls() =
        runBlocking {
            val tool =
                AppFunctionMetadata(
                    id = "com.example.chat.sendMessage",
                    packageName = "com.example.chat",
                    isEnabled = true,
                    schema = null,
                    description = "Sends a chat message",
                    parameters =
                        listOf(
                            AppFunctionParameterMetadata(
                                name = "text",
                                isRequired = true,
                                dataType = AppFunctionStringTypeMetadata(isNullable = false),
                                description = "Message body",
                            ),
                        ),
                    response =
                        AppFunctionResponseMetadata(
                            valueType = AppFunctionStringTypeMetadata(isNullable = false),
                            description = "",
                        ),
                    components = AppFunctionComponentsMetadata(emptyMap()),
                )

            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("Please sendMessage to Alice"),
                    tools = listOf(tool),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                ) as LlmResponse.Success

            val toolCalls = response.parts.filterIsInstance<LlmResponsePart.ToolCall>()
            assertEquals(1, toolCalls.size)
            assertEquals("com.example.chat.sendMessage", toolCalls[0].functionId)
            assertEquals("com.example.chat", toolCalls[0].packageName)
        }

    @Test
    fun generateResponse_toolResponse_returnsProcessedText() =
        runBlocking {
            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input =
                        LlmInput.ToolResponse(
                            listOf(
                                ToolOutput(
                                    functionId = "com.example.chat.sendMessage",
                                    callId = "call456",
                                    result = "{\"sent\":true}",
                                ),
                            ),
                        ),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                ) as LlmResponse.Success

            val textParts = response.parts.filterIsInstance<LlmResponsePart.Text>()
            assertTrue(textParts.any { it.text.contains("sendMessage") })
        }

    @Test
    fun normalizeToolArguments_gmsDeviceStateMetadata_unwrapsPropertiesAndMapsParams() {
        val tool =
            createDeviceStateTool(
                id = "android.gms.appfunctions.GmsDeviceState#getDeviceStateMetadata",
                paramName = "getDeviceStateMetadataParams",
            )
        val rawArgs =
            mapOf(
                "properties" to
                    mapOf(
                        "deviceStateMetadata" to mapOf("requestInitiatedWhileUnlocked" to false),
                    ),
            )

        val normalized = provider.normalizeToolArguments(tool, rawArgs)

        assertTrue(normalized.containsKey("getDeviceStateMetadataParams"))
        @Suppress("UNCHECKED_CAST")
        val params = normalized["getDeviceStateMetadataParams"] as Map<String, Any?>
        assertEquals(true, params["requestInitiatedWhileUnlocked"])
    }

    @Test
    fun normalizeToolArguments_gmsDeviceState_flattenedArguments() {
        val tool =
            createDeviceStateTool(
                id = "android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState",
                paramName = "getUncategorizedDeviceStateParams",
            )
        val rawArgs = mapOf("requestInitiatedWhileUnlocked" to true)

        val normalized = provider.normalizeToolArguments(tool, rawArgs)

        assertTrue(normalized.containsKey("getUncategorizedDeviceStateParams"))
        @Suppress("UNCHECKED_CAST")
        val params = normalized["getUncategorizedDeviceStateParams"] as Map<String, Any?>
        assertEquals(true, params["requestInitiatedWhileUnlocked"])
    }

    @Test
    fun normalizeToolArguments_gmsDeviceStateItem_resolvesCanonicalKey() {
        val tool =
            createDeviceStateTool(
                id = "android.gms.appfunctions.GmsDeviceState#getDeviceStateItem",
                paramName = "getDeviceStateItemParams",
            )
        val rawArgs =
            mapOf(
                "properties" to
                    mapOf(
                        "deviceStateItem" to
                            mapOf(
                                "key" to listOf("protection", "_share_visibility"),
                            ),
                    ),
            )

        val normalized = provider.normalizeToolArguments(tool, rawArgs)

        assertTrue(normalized.containsKey("getDeviceStateItemParams"))
        @Suppress("UNCHECKED_CAST")
        val params = normalized["getDeviceStateItemParams"] as Map<String, Any?>
        assertEquals("quick_share_visibility", params["key"])
    }

    @Test
    fun matchTool_uncategorizedState_matchesGetUncategorizedDeviceState() {
        val tool =
            createDeviceStateTool(
                id = "android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState",
                paramName = "getUncategorizedDeviceStateParams",
            )
        val matched = provider.matchTool("getUncategorizedState", listOf(tool))
        assertNotNull(matched)
        assertEquals("android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState", matched?.id)
    }

    @Test
    fun generateResponse_deviceState_matchesToolAndGeneratesParams() =
        runBlocking {
            val tool =
                createDeviceStateTool(
                    id = "android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState",
                    paramName = "getUncategorizedDeviceStateParams",
                )

            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("show device state"),
                    tools = listOf(tool),
                    apiKey = "",
                    modelName = "gemma-4-e2b",
                ) as LlmResponse.Success

            val toolCalls = response.parts.filterIsInstance<LlmResponsePart.ToolCall>()
            assertEquals(1, toolCalls.size)
            assertEquals("android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState", toolCalls[0].functionId)
            assertTrue(toolCalls[0].arguments.containsKey("getUncategorizedDeviceStateParams"))
        }

    @Test
    fun generateResponse_deviceStateResult_formatsSummary() =
        runBlocking {
            val deviceStateJson =
                """
                {
                  "perScreenDeviceStates": [
                    {
                      "description": "Theft protection",
                      "deviceStateItems": [
                        {
                          "key": "theft_detection_lock_setting",
                          "name": { "english": "Theft Detection Lock" },
                          "jsonValue": "false"
                        }
                      ]
                    },
                    {
                      "description": "Quick Share",
                      "deviceStateItems": [
                        {
                          "key": "quick_share_visibility",
                          "name": { "english": "Receive with Quick Share" },
                          "jsonValue": "HIDDEN"
                        }
                      ]
                    }
                  ]
                }
                """.trimIndent()

            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input =
                        LlmInput.ToolResponse(
                            listOf(
                                ToolOutput(
                                    functionId = "android.gms.appfunctions.GmsDeviceState#getUncategorizedDeviceState",
                                    callId = "call789",
                                    result = deviceStateJson,
                                ),
                            ),
                        ),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e2b",
                ) as LlmResponse.Success

            val textParts = response.parts.filterIsInstance<LlmResponsePart.Text>()
            assertEquals(1, textParts.size)
            val text = textParts[0].text
            assertTrue(text.contains("Theft protection"))
            assertTrue(text.contains("Theft Detection Lock: Disabled"))
            assertTrue(text.contains("Quick Share"))
            assertTrue(text.contains("Receive with Quick Share: Hidden"))
        }

    private fun createDeviceStateTool(
        id: String,
        paramName: String,
    ): AppFunctionMetadata {
        val objectType =
            AppFunctionObjectTypeMetadata(
                properties =
                    mapOf(
                        "requestInitiatedWhileUnlocked" to AppFunctionBooleanTypeMetadata(isNullable = false),
                    ),
                required = listOf("requestInitiatedWhileUnlocked"),
                qualifiedName = null,
                isNullable = false,
            )
        return AppFunctionMetadata(
            id = id,
            packageName = "com.google.android.gms",
            isEnabled = true,
            schema = null,
            description = "Gets device state",
            parameters =
                listOf(
                    AppFunctionParameterMetadata(
                        name = paramName,
                        isRequired = true,
                        dataType = objectType,
                        description = "Parameters",
                    ),
                ),
            response =
                AppFunctionResponseMetadata(
                    valueType = AppFunctionStringTypeMetadata(isNullable = false),
                    description = "",
                ),
            components = AppFunctionComponentsMetadata(emptyMap()),
        )
    }

    @Test
    fun normalizeToolArguments_searchContacts_omittedQuery_extractsFromUserQuery() {
        val tool = createSearchContactsTool()
        val rawArgs = mapOf("contactType" to "INDIVIDUAL")

        val normalized = provider.normalizeToolArguments(tool, rawArgs, userQuery = "call david")

        assertEquals("INDIVIDUAL", normalized["contactType"])
        assertEquals("david", normalized["query"])
    }

    @Test
    fun normalizeToolArguments_searchContacts_nameAlias_mappedToQuery() {
        val tool = createSearchContactsTool()
        val rawArgs = mapOf("name" to "david", "contactType" to "INDIVIDUAL")

        val normalized = provider.normalizeToolArguments(tool, rawArgs)

        assertEquals("INDIVIDUAL", normalized["contactType"])
        assertEquals("david", normalized["query"])
    }

    @Test
    fun normalizeToolArguments_makeCall_resolvesFromPreviousSearchContacts() {
        val tool =
            AppFunctionMetadata(
                id = "com.example.chatapp.appfunctions.BaseChatAppFunctionService#makeCall",
                packageName = "com.example.chatapp",
                isEnabled = true,
                schema = null,
                description = "Makes a phone call",
                parameters =
                    listOf(
                        AppFunctionParameterMetadata(
                            name = "endpointValue",
                            isRequired = true,
                            dataType = AppFunctionStringTypeMetadata(isNullable = false),
                            description = "Endpoint value",
                        ),
                    ),
                response =
                    AppFunctionResponseMetadata(
                        valueType = AppFunctionStringTypeMetadata(isNullable = false),
                        description = "",
                    ),
                components = AppFunctionComponentsMetadata(emptyMap()),
            )
        val history =
            listOf(
                GemmaTurn.User("call david"),
                GemmaTurn.ToolResult(
                    listOf(
                        ToolOutput(
                            functionId = "com.example.chatapp.appfunctions.BaseChatAppFunctionService#searchContacts",
                            callId = "call1",
                            result = "[{\"endpointValue\":\"david_id_123\",\"contactDisplayName\":\"David\"}]",
                        ),
                    ),
                ),
            )

        val normalized = provider.normalizeToolArguments(tool, emptyMap(), userQuery = "", history = history)

        assertEquals("david_id_123", normalized["endpointValue"])
    }

    private fun createSearchContactsTool(): AppFunctionMetadata {
        return AppFunctionMetadata(
            id = "com.example.chatapp.appfunctions.BaseChatAppFunctionService#searchContacts",
            packageName = "com.example.chatapp",
            isEnabled = true,
            schema = null,
            description = "Searches contacts",
            parameters =
                listOf(
                    AppFunctionParameterMetadata(
                        name = "query",
                        isRequired = true,
                        dataType = AppFunctionStringTypeMetadata(isNullable = false),
                        description = "Query",
                    ),
                    AppFunctionParameterMetadata(
                        name = "contactType",
                        isRequired = true,
                        dataType = AppFunctionStringTypeMetadata(isNullable = false),
                        description = "Contact type",
                    ),
                ),
            response =
                AppFunctionResponseMetadata(
                    valueType = AppFunctionStringTypeMetadata(isNullable = false),
                    description = "",
                ),
            components = AppFunctionComponentsMetadata(emptyMap()),
        )
    }
}
