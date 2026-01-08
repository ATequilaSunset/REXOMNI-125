    def _generate_transformers_batch(
        self, batch_messages: List[List[Dict]], batch_images: List[Image.Image]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate using Transformers model for batch processing"""

        # Prepare batch inputs
        batch_texts = []
        for messages in batch_messages:
            #batch_messages：字典和列表的复杂混合体，包含了rgb格式的原始图片和text文本prompt
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)
            #batch_texts主要内容 = '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Detect man, woman, yellow flower, sofa, robot-shope light, blanket, microwave, laptop, cup, white chair, lamp. Output the bounding box coordinates in [x0, y0, x1, y1] format.<|im_end|>\n<|im_start|>assistant\n'



        # Process inputs for batch
        generation_start = time.time()

        #--------------------------------------------------------------------------------------
        #关键还是在这里！
        #送入文本和图片，加载成token，而不仅仅是张量。
        #所以，计划是这样的：
        '''
        1. self.processor只送入batch_text,不送batch_images。拿到的结果为一个字典,包含两个key:
            ① input_ids(形状：(batch_size, sequence_length)) —————————————— 整数 token ID 序列，每个token整数对应词表中的一个索引。
                sequence_lengh的具体内容：[0, 0, 0, ..., 0, 15496, 2159, 389, 345, 30, x, 50256, ...]——————————前面是图像占位符（全0），中间是文本，后面是填充。
                对于图像占位符：模型配置文件中定义了最大视觉 token 数量，无论实际图像大小，占位符数量不变，但 image_grid_thw（形状(batch_size, 3)其中3代表T帧数/H高度方向patches数/W宽度方向patches数） 告诉模型哪些位置是有效图像 token
            ② attention_mask(形状与inputs_ids一致(batch_size, sequence_length))————————————作用：告诉模型 "哪些 token 是真实内容，哪些是填充（padding）"
        2. 单独使用一个VIT或者CNN将特征图卷成token的形式,然后插入到step1中得到的input_ids中。
        3. 再看需不需要修改attention_mask
        
        '''
        
        inputs = self.processor( #把标准batch_texts和标准rgb图片都 转换成 张量形式
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # inputs_control = self.processor( #把标准batch_texts和标准rgb图片都 转换成 张量形式
        #     text=batch_texts,
        #     padding=True,
        #     return_tensors="pt",
        # ).to(self.model.device)

        # if inputs[input_ids]==inputs_control[input_ids] :#用于验证：加/不加图片输入，生成的Token序列有没有区别
        #     print("yes\n") 


        # #小实验：把字典中pixel_values张量换成非三维的张量形式。已知这个pixel_values张量后续是要送入VIT得到token向量然后和文本Token向量合并的。
        # temp = torch.rand(128,224,224)
        # inputs[pixel_values] = temp
        # #理论上应该不行，因为VIT对第一个维度的值固定要求是3
        # #所以，只能把里面的源码里面"VIT固定要求维度是3"的位置改成“不固定要求”————而这也就意味着要改VIT的CNN层，因为CNN会硬性规定通道数为3,例如：
        # '''
        # # 标准的Conv2d层
        # conv = nn.Conv2d(
        #     in_channels=3,      # 期望输入3通道
        #     out_channels=64,
        #     kernel_size=7
        # )
        # '''
        


        # #--------------------------------------------------------------------------------------

        print("===========================================================")
        for key, value in inputs.items():
            print(f"{key}: {value.shape}")
        print("===========================================================")


        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Generate for entire batch
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs ,return_dict_in_generate=True, output_scores=True)

        generation_time = time.time() - generation_start

        # Decode batch results
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        # Prepare generation info for each item
        batch_generation_info = []
        for i, output_ids in enumerate(generated_ids_trimmed):
            num_output_tokens = len(output_ids)
            num_prompt_tokens = len(inputs.input_ids[i])
            tokens_per_second = (
                num_output_tokens / generation_time if generation_time > 0 else 0
            )

            generation_info = {
                "num_output_tokens": num_output_tokens,
                "num_prompt_tokens": num_prompt_tokens,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
            }
            batch_generation_info.append(generation_info)

        return batch_outputs, batch_generation_info
