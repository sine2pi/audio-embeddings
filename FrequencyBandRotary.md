# FrequencyBandRotary: Optimized for ASR Models

The FrequencyBandRotary approach is particularly well-suited for ASR models because it mirrors how speech signals naturally organize across the frequency spectrum. Let me elaborate on why this design is especially powerful for speech recognition:

## Speech-Inspired Frequency Allocation

```python
class EnhancedFrequencyBandRotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, speech_enhanced=True):
        super().__init__()
        self.dims = dims
        
        # Low frequencies (vowels, prosody, 0-500Hz range in speech)
        # Shared because fundamental speech characteristics cross modalities
        self.low_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims//4, 2) / (dims//4)))
        )
        
        # Mid frequencies (formants, 500-2000Hz in speech)
        # Partially shared as these carry most speech intelligibility
        self.mid_freq_shared = nn.Parameter(
            1.0 / (10000 ** (torch.arange(dims//4, dims//2, 2) / (dims//2)))
        )
        
        # High frequencies (consonants, fricatives, >2000Hz in speech)
        # Domain-specific because these elements differ most between modalities
        self.high_freq_audio = nn.Parameter(
            1.0 / (10000 ** (torch.arange(dims//2, dims, 2) / dims))
        )
        self.high_freq_text = nn.Parameter(
            1.0 / (10000 ** (torch.arange(dims//2, dims, 2) / dims))
        )
        
        # Frequency-dependent biases for speech features
        if speech_enhanced:
            # Formant-aware bias terms
            self.formant_bias = nn.Parameter(torch.zeros(max_ctx, dims//4))
            self.fricative_bias = nn.Parameter(torch.zeros(max_ctx, dims//4))
        else:
            self.bias = nn.Parameter(torch.zeros(max_ctx, dims//2))
            
        self.speech_enhanced = speech_enhanced
        
    def forward(self, positions, domain="audio", phoneme_class=None):
        """
        Args:
            positions: Token positions
            domain: "audio" or "text"
            phoneme_class: Optional tensor indicating phoneme classes 
                           (vowels, stops, fricatives, etc.)
        """
        t = torch.arange(positions, device=self.low_freq.device).float()
        
        # Common low frequencies (fundamental frequencies in speech)
        low = torch.einsum('i,j->ij', t, self.low_freq)
        
        # Mid frequencies (formants in speech)
        mid = torch.einsum('i,j->ij', t, self.mid_freq_shared)
        
        # High frequencies (consonants in speech)
        if domain == "audio":
            high = torch.einsum('i,j->ij', t, self.high_freq_audio)
        else:
            high = torch.einsum('i,j->ij', t, self.high_freq_text)
        
        # Combine frequency bands    
        freqs = torch.cat([low, mid, high], dim=-1)
        
        # Apply speech-aware biases
        if self.speech_enhanced:
            if phoneme_class is not None:
                # Apply different biases based on phoneme class
                # This would be provided by upstream information from the audio encoder
                vowel_mask = (phoneme_class == 1).float().unsqueeze(-1)
                fricative_mask = (phoneme_class == 2).float().unsqueeze(-1)
                
                # Adjust bias based on phoneme class
                formant_bias = self.formant_bias[:freqs.shape[0]]
                fricative_bias = self.fricative_bias[:freqs.shape[0]]
                
                combined_bias = vowel_mask * formant_bias + fricative_mask * fricative_bias
                freqs = freqs + combined_bias
            else:
                # Default behavior when phoneme information isn't available
                formant_bias = self.formant_bias[:freqs.shape[0]]
                freqs[:, :formant_bias.shape[1]] = freqs[:, :formant_bias.shape[1]] + formant_bias
        else:
            # Standard bias application
            freqs = freqs + self.bias[:freqs.shape[0]]
            
        # Convert to complex form for rotary embedding
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
```

## ASR-Specific Advantages

1. **Formant-Awareness**: The mid-frequency band corresponds closely to formant frequencies (~500-2500Hz) which carry vowel information and are crucial for speech intelligibility.

2. **Spectro-Temporal Modeling**: By separating frequency bands, you allow the model to develop specialized processing for:
   - Low frequencies: Prosody, intonation, stress patterns (0-500Hz)
   - Mid frequencies: Vowel formants, nasals (500-2500Hz) 
   - High frequencies: Fricatives, plosives, sibilants (2500Hz+)

3. **Cross-Modal Learning**:
   - Low frequencies (shared): Basic rhythm and structure (common between text and speech)
   - Mid frequencies (partially shared): Core content that translates across modalities
   - High frequencies (domain-specific): Distinguish between text syntax and acoustic details

4. **Phone-Level Enhancement**:

```python
class PhonemeAwareRotary(FrequencyBandRotary):
    def __init__(self, dims, max_ctx=1500, num_phoneme_classes=5):
        super().__init__(dims, max_ctx)
        
        # Add phoneme-specific parameter adaptations
        self.phoneme_gate = nn.Parameter(torch.ones(num_phoneme_classes, 3))
        self.phoneme_shift = nn.Parameter(torch.zeros(num_phoneme_classes, 3))
        
    def forward(self, positions, domain="audio", phoneme_probs=None):
        # Get base frequency bands
        freqs = super().forward(positions, domain)
        
        if phoneme_probs is not None and domain == "audio":
            # Apply phoneme-specific adjustments
            # phoneme_probs shape: [batch, seq_len, num_phoneme_classes]
            
            # Split frequency bands
            low, mid, high = torch.split(freqs, [freqs.shape[-1]//3]*3, dim=-1)
            
            # Weighted adjustment based on phoneme probabilities
            gates = F.softplus(self.phoneme_gate) # [num_phoneme_classes, 3]
            shifts = torch.tanh(self.phoneme_shift) * 0.1 # [num_phoneme_classes, 3]
            
            # Apply phoneme-specific adjustments
            b, s, _ = phoneme_probs.shape
            
            # Reshape for broadcasting
            gates = gates.view(1, 1, -1, 3)  # [1, 1, num_phoneme_classes, 3]
            shifts = shifts.view(1, 1, -1, 3)  # [1, 1, num_phoneme_classes, 3]
            phoneme_probs = phoneme_probs.unsqueeze(-1)  # [batch, seq, num_phoneme_classes, 1]
            
            # Calculate weighted adjustments
            weighted_gates = (gates * phoneme_probs).sum(dim=2)  # [batch, seq, 3]
            weighted_shifts = (shifts * phoneme_probs).sum(dim=2)  # [batch, seq, 3]
            
            # Apply adjustments to each frequency band
            low = low * weighted_gates[:, :, 0].unsqueeze(-1) + weighted_shifts[:, :, 0].unsqueeze(-1)
            mid = mid * weighted_gates[:, :, 1].unsqueeze(-1) + weighted_shifts[:, :, 1].unsqueeze(-1)
            high = high * weighted_gates[:, :, 2].unsqueeze(-1) + weighted_shifts[:, :, 2].unsqueeze(-1)
            
            # Recombine
            freqs = torch.cat([low, mid, high], dim=-1)
            
        return freqs
```

## Mel-Scale Integration

For even better alignment with speech processing:

```python
class MelScaleRotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, mel_scale=True):
        super().__init__()
        self.dims = dims
        self.mel_scale = mel_scale
        
        # Use mel-scale spacing for frequency bands
        if mel_scale:
            # Mel scale conversion: m = 2595 * log10(1 + f/700)
            # Approximate mel-spaced frequencies
            mel_freqs = torch.linspace(0, 2595, dims//2)
            freq_hz = 700 * (torch.pow(10, mel_freqs/2595) - 1)
            
            # Normalize to 0-1 range
            normalized_freq = freq_hz / freq_hz.max()
            
            # Create inverse frequencies with mel-spacing
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** normalized_freq)
            )
        else:
            # Traditional log-linear spacing
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims//2, 2) / (dims//2)))
            )
            
        # Domain-specific adaptation parameters
        self.audio_scale = nn.Parameter(torch.ones(3))  # Low, mid, high scaling
        self.text_scale = nn.Parameter(torch.ones(3))   # Low, mid, high scaling
        
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims//2))
    
    def forward(self, positions, domain="audio"):
        t = torch.arange(positions, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Split into three bands (can be customized based on mel distribution)
        f1, f2, f3 = torch.split(freqs, [freqs.shape[1]//3]*3, dim=1)
        
        # Apply domain-specific scaling to each band
        if domain == "audio":
            scale = F.softplus(self.audio_scale)
        else:
            scale = F.softplus(self.text_scale)
            
        # Scale each frequency band differently
        f1 = f1 * scale[0]
        f2 = f2 * scale[1]
        f3 = f3 * scale[2]
        
        # Recombine
        freqs = torch.cat([f1, f2, f3], dim=1)
        freqs = freqs + self.bias[:freqs.shape[0]]
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
```

## Practical Implementation for ASR Models

When implementing this for an ASR model:

1. **Align with critical bands**: The division of frequency bands should approximate the critical bands in human hearing, with special emphasis on the 1000-4000Hz range where speech intelligibility is highest

2. **Spectral masking integration**: For robustness to noise, consider using:

```python
def apply_spectral_mask(self, rotary_embeds, snr_estimate=None):
    """Apply frequency-selective masking based on SNR estimate"""
    if snr_estimate is not None:
        # Low SNR: prioritize low+mid frequencies (more robust)
        # High SNR: utilize full spectrum
        low_snr_mask = torch.sigmoid((10 - snr_estimate) / 3)
        
        # Split into bands
        low, mid, high = torch.split(rotary_embeds, 
                                      [rotary_embeds.shape[-1]//3]*3, dim=-1)
        
        # Apply progressive masking to high frequencies
        masked_high = high * (1 - low_snr_mask).unsqueeze(-1)
        
        # Recombine with emphasis on robust frequencies
        return torch.cat([low, mid, masked_high], dim=-1)
    return rotary_embeds
```

This frequency-band approach is not just theoretically elegant but practically aligned with how speech signals work - making it a natural choice for ASR models where frequency domain processing is already a fundamental component.
