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
package com.example.appfunctions.agent

import android.app.Application
import android.util.Log
import dagger.hilt.android.HiltAndroidApp
import java.io.File

@HiltAndroidApp
class AppFunctionsApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        loadOpenClLibrary()
    }

    private fun loadOpenClLibrary() {
        val openClPaths = listOf(
            "/vendor/lib64/libOpenCL.so",
            "/vendor/lib64/libOpenCL_adreno.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/system/lib64/libOpenCL.so",
        )
        for (path in openClPaths) {
            val file = File(path)
            if (file.exists()) {
                try {
                    System.load(path)
                    Log.i(TAG, "Successfully loaded OpenCL library from $path")
                    return
                } catch (t: Throwable) {
                    Log.w(TAG, "Attempt to load $path failed: ${t.message}")
                }
            }
        }
        try {
            System.loadLibrary("OpenCL")
            Log.i(TAG, "Successfully loaded OpenCL library via System.loadLibrary")
        } catch (t: Throwable) {
            Log.w(TAG, "Attempt to load OpenCL via System.loadLibrary failed: ${t.message}")
        }
    }

    companion object {
        private const val TAG = "AppFunctionsApp"
    }
}
