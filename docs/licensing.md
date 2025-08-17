# Licensing Guide

## Overview

Dream-CAD is licensed under the MIT License, but individual models have their own licensing requirements. This guide clarifies the licensing situation for each component.

## Dream-CAD Framework License

**License**: MIT License  
**Commercial Use**: ✅ Allowed  
**Modification**: ✅ Allowed  
**Distribution**: ✅ Allowed  
**Private Use**: ✅ Allowed  

The Dream-CAD framework itself (all code in this repository) is MIT licensed and free for any use.

## Individual Model Licenses

### MVDream
**License**: Apache 2.0  
**Commercial Use**: ✅ Allowed  
**Key Points**:
- Free for commercial and non-commercial use
- Must include license notice in distributions
- Must state changes made to the original code
- No trademark use rights

**License URL**: https://github.com/bytedance/MVDream/blob/main/LICENSE

### TripoSR
**License**: MIT License  
**Commercial Use**: ✅ Allowed  
**Key Points**:
- Completely free for any use
- No attribution required (but appreciated)
- Can be used in proprietary software
- No warranty provided

**License URL**: https://github.com/VAST-AI-Research/TripoSR/blob/main/LICENSE

### Stable-Fast-3D
**License**: Stability AI Community License  
**Commercial Use**: ⚠️ Conditional  
**Key Points**:
- Free for research and non-commercial use
- Commercial use requires agreement with Stability AI
- Revenue threshold may apply
- Contact Stability AI for commercial licensing

**License URL**: Check model card on HuggingFace

### TRELLIS
**License**: Apache 2.0  
**Commercial Use**: ✅ Allowed  
**Key Points**:
- Free for commercial and non-commercial use
- Microsoft Research project
- Must include license notice
- Patent grants included

**License URL**: https://github.com/microsoft/TRELLIS/blob/main/LICENSE

### Hunyuan3D-Mini
**License**: Tencent Hunyuan License  
**Commercial Use**: ⚠️ Conditional  
**Key Points**:
- **Free for**: Projects with <1M monthly active users
- **Paid license required**: >1M MAU or >$1M annual revenue
- **Restrictions**: Cannot use for illegal content generation
- **Attribution**: Required in commercial products

**License Details**:
```
Revenue/MAU Thresholds:
- < 1M MAU: Free
- > 1M MAU: Commercial license required
- > $1M revenue: Commercial license required
```

**Contact**: hunyuan3d@tencent.com for commercial licensing

## Usage Scenarios

### Personal/Hobby Projects
✅ **All models allowed** - No restrictions for personal use

### Academic Research
✅ **All models allowed** - All models permit research use

### Open Source Projects
✅ **Allowed models**:
- MVDream (Apache 2.0)
- TripoSR (MIT)
- TRELLIS (Apache 2.0)

⚠️ **Check license**:
- Stable-Fast-3D
- Hunyuan3D-Mini

### Small Commercial Projects (<1M users)
✅ **Allowed models**:
- MVDream
- TripoSR
- TRELLIS
- Hunyuan3D-Mini (under threshold)

⚠️ **Requires license**:
- Stable-Fast-3D (contact Stability AI)

### Large Commercial Projects (>1M users)
✅ **Allowed models**:
- MVDream
- TripoSR
- TRELLIS

💰 **Requires paid license**:
- Stable-Fast-3D
- Hunyuan3D-Mini

## Compliance Checklist

### Before Using in Production

1. **Identify your use case**:
   - [ ] Personal/hobby
   - [ ] Academic research
   - [ ] Open source project
   - [ ] Commercial (specify scale)

2. **Check user/revenue metrics**:
   - [ ] Monthly active users estimate
   - [ ] Annual revenue projection
   - [ ] Growth trajectory

3. **Review each model's license**:
   - [ ] Read full license text
   - [ ] Check commercial use terms
   - [ ] Identify attribution requirements
   - [ ] Note any restrictions

4. **Obtain necessary licenses**:
   - [ ] Contact vendors for commercial licenses
   - [ ] Get written agreements
   - [ ] Budget for license fees

5. **Implement compliance**:
   - [ ] Add required attributions
   - [ ] Include license notices
   - [ ] Track usage metrics
   - [ ] Set up alerts for threshold crossing

## Attribution Requirements

### MVDream Attribution
```
This product uses MVDream by ByteDance
Licensed under Apache 2.0
https://github.com/bytedance/MVDream
```

### TripoSR Attribution (Optional but appreciated)
```
3D generation powered by TripoSR
https://github.com/VAST-AI-Research/TripoSR
```

### TRELLIS Attribution
```
This product includes TRELLIS by Microsoft Research
Licensed under Apache 2.0
https://github.com/microsoft/TRELLIS
```

### Hunyuan3D Attribution (Required for commercial use)
```
3D models generated using Hunyuan3D by Tencent
https://github.com/tencent/Hunyuan3D
```

## License Compatibility

### MIT + Apache 2.0
✅ **Compatible** - Can be combined freely

### MIT + Proprietary
⚠️ **Check terms** - Depends on proprietary license

### Apache 2.0 + GPL
⚠️ **Complex** - Requires careful handling

## Model Weight Licenses

### Pre-trained Weight Considerations

Models may use pre-trained weights with additional restrictions:

1. **Stable Diffusion weights**: 
   - CreativeML Open RAIL-M license
   - Allows commercial use with restrictions
   - No illegal content generation

2. **CLIP weights**:
   - MIT License
   - Free for any use

3. **Custom trained weights**:
   - Check individual model cards
   - May have additional restrictions

## Frequently Asked Questions

### Q: Can I use Dream-CAD for commercial products?
**A**: The Dream-CAD framework is MIT licensed and free for commercial use. Individual models have their own licenses - check each model's terms.

### Q: Do I need to pay for Hunyuan3D if my app has 500K users?
**A**: No, the free tier covers up to 1M monthly active users.

### Q: Can I modify the models?
**A**: Depends on the model:
- ✅ TripoSR (MIT) - Yes
- ✅ MVDream (Apache) - Yes, with attribution
- ✅ TRELLIS (Apache) - Yes, with attribution
- ⚠️ Others - Check specific license

### Q: Can I use generated models in a game I'm selling?
**A**: Generally yes, but:
- Check each model's license
- Hunyuan3D requires license if game has >1M MAU
- Stable-Fast-3D may require commercial agreement

### Q: What if I exceed the usage threshold mid-project?
**A**: 
1. Monitor usage metrics regularly
2. Contact vendors before crossing thresholds
3. Budget for potential licensing costs
4. Consider switching models if needed

### Q: Can I use these models for NFT generation?
**A**: Check each model's terms:
- Some explicitly prohibit NFT/crypto use
- Others may require special licensing
- Get written permission when unclear

## Legal Disclaimers

**This guide is for informational purposes only and does not constitute legal advice.**

Key points:
1. Always read the full license text
2. When in doubt, contact the model creators
3. Keep records of license agreements
4. Monitor your usage metrics
5. Budget for potential license fees
6. Consider legal consultation for large projects

## Contact Information

### Model Licensing Contacts

**Stability AI (Stable-Fast-3D)**:
- Website: https://stability.ai
- Email: Check website for current contact

**Tencent (Hunyuan3D)**:
- Email: hunyuan3d@tencent.com
- Website: https://hunyuan.tencent.com

**Microsoft (TRELLIS)**:
- GitHub: https://github.com/microsoft/TRELLIS/issues

**ByteDance (MVDream)**:
- GitHub: https://github.com/bytedance/MVDream/issues

## License Change Monitoring

Models may update their licenses. To stay compliant:

1. **Subscribe to notifications**:
   ```bash
   # Watch GitHub repositories
   gh repo fork MODEL_REPO --watch
   ```

2. **Regular audits**:
   - Quarterly license review
   - Check model cards for updates
   - Review vendor announcements

3. **Version pinning**:
   ```python
   # Pin to specific versions with known licenses
   model_versions = {
       "triposr": "v1.0.0",
       "mvdream": "v1.2.0",
   }
   ```

## Summary Table

| Model | License | Commercial Use | Attribution | Revenue Limit | Contact Required |
|-------|---------|---------------|-------------|---------------|------------------|
| Dream-CAD Framework | MIT | ✅ Free | Optional | None | No |
| MVDream | Apache 2.0 | ✅ Free | Required | None | No |
| TripoSR | MIT | ✅ Free | Optional | None | No |
| Stable-Fast-3D | Community | ⚠️ Conditional | Required | Varies | Yes |
| TRELLIS | Apache 2.0 | ✅ Free | Required | None | No |
| Hunyuan3D-Mini | Proprietary | ⚠️ Conditional | Required | 1M MAU | If exceeded |

## Conclusion

Understanding licensing is crucial for commercial deployment:
1. **Most permissive**: TripoSR (MIT)
2. **Good for most uses**: MVDream, TRELLIS (Apache 2.0)
3. **Check before commercial use**: Stable-Fast-3D, Hunyuan3D-Mini
4. **Always read the full license**
5. **When in doubt, ask the creators**