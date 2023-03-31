package io.github.zap8600.aimobileapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.viewModels
import androidx.databinding.DataBindingUtil
import io.github.zap8600.aimobileapp.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private val gpt2: io.github.zap8600.aimobileapp.ml.GPT2Client by viewModels()
    private val codegen: io.github.zap8600.aimobileapp.ml.CodegenClient by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding: ActivityMainBinding
                = DataBindingUtil.setContentView(this, R.layout.activity_main)
        
        binding.vm = gpt2
        // binding.vm = codegen

        binding.lifecycleOwner = this
    }
}
