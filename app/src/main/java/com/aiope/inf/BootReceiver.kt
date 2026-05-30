package com.aiope.inf

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            val serviceIntent = Intent(context, InferenceService::class.java).apply {
                action = InferenceService.ACTION_START_SERVER
            }
            context.startForegroundService(serviceIntent)
        }
    }
}
