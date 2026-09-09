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
import androidx.appfunctions.metadata.AppFunctionComponentsMetadata
import androidx.appfunctions.metadata.AppFunctionMetadata
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

class Gemma4E4bProviderImplTest {
    private val toolConverter = GeminiToolConverter()
    private lateinit var provider: Gemma4E4bProviderImpl

    @Before
    fun setUp() {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.i(any(), any()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        provider = Gemma4E4bProviderImpl(toolConverter)
    }

    @Test
    fun generateResponse_userMessage_returnsGemmaResponse() =
        runBlocking {
            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input = LlmInput.UserMessage("Hello Gemma 4 E4B"),
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
                )

            assertTrue(response is LlmResponse.Success)
            val success = response as LlmResponse.Success
            val toolCallParts = success.parts.filterIsInstance<LlmResponsePart.ToolCall>()
            assertTrue("Expected a ToolCall part", toolCallParts.isNotEmpty())
            assertEquals("com.example.chat.sendMessage", toolCallParts.first().functionId)
        }

    @Test
    fun generateResponse_toolResponse_formatsSummary() =
        runBlocking {
            val response =
                provider.generateResponse(
                    previousInteractionId = null,
                    input =
                        LlmInput.ToolResponse(
                            outputs =
                                listOf(
                                    ToolOutput(
                                        functionId = "com.example.chat.sendMessage",
                                        result = "{\"sent\":true}",
                                        callId = "call-1",
                                    ),
                                ),
                        ),
                    tools = emptyList(),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                )

            assertTrue(response is LlmResponse.Success)
            val success = response as LlmResponse.Success
            val textParts = success.parts.filterIsInstance<LlmResponsePart.Text>()
            assertTrue(textParts.isNotEmpty())
            assertTrue(textParts.any { it.text.contains("sendMessage") })
        }

    @Test
    fun generateResponse_makeCall_returnsToolCall() =
        runBlocking {
            val tool =
                AppFunctionMetadata(
                    id = "com.example.telecom.makeCall",
                    packageName = "com.example.telecom",
                    isEnabled = true,
                    schema = null,
                    description = "Makes a phone call",
                    parameters =
                        listOf(
                            AppFunctionParameterMetadata(
                                name = "endpointValue",
                                isRequired = true,
                                dataType = AppFunctionStringTypeMetadata(isNullable = false),
                                description = "Phone number or contact",
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
                    input = LlmInput.UserMessage("Call Bob"),
                    tools = listOf(tool),
                    apiKey = "",
                    modelName = "gemma-4-e4b",
                )

            assertTrue(response is LlmResponse.Success)
            val success = response as LlmResponse.Success
            val toolCallParts = success.parts.filterIsInstance<LlmResponsePart.ToolCall>()
            assertTrue("Expected a ToolCall part", toolCallParts.isNotEmpty())
            assertEquals("com.example.telecom.makeCall", toolCallParts.first().functionId)
        }
}
