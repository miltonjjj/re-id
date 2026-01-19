"""
VLM API 连接测试 - 极简版
只检测 VLM 是否可用，只输出错误原因
"""

import base64
import os
from openai import OpenAI


def load_image_as_base64(image_path):
    """加载图像并转换为base64"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    return base64.b64encode(img_bytes).decode('utf-8')


def test_vlm_connection():
    """
    测试 VLM API 是否可用
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    print("\n" + "="*70)
    print("VLM API 连接测试")
    print("="*70)
    
    # API配置
    API_KEY = "sk-zk24390385d11aba6430c32a49e645dc3ee695b79880a6ab"
    BASE_URL = "https://api.zhizengzeng.com/v1/"
    MODEL = "gpt-4o-mini"
    
    # 测试图片路径
    test_image_path = "/data/mingye/re-id/data/voronoi_processed_wo_vlm/Adrian_ep4/paths.png"
    
    print(f"\n配置: {BASE_URL} | {MODEL}")
    print(f"正在测试...\n")
    
    try:
        # 加载测试图片
        if not os.path.exists(test_image_path):
            print(f"✗ 测试图片不存在: {test_image_path}")
            return False
        
        test_img = load_image_as_base64(test_image_path)
        
        # 创建客户端
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # 发送测试请求
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{test_img}"}
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.2
        )
        
        # 先检查 error 字段
        if hasattr(response, 'error') and response.error:
            error_info = response.error
            error_msg = error_info.get('message', 'Unknown') if isinstance(error_info, dict) else str(error_info)
            error_type = error_info.get('type', 'Unknown') if isinstance(error_info, dict) else 'Unknown'
            error_code = error_info.get('code', 'N/A') if isinstance(error_info, dict) else 'N/A'
            
            print(f"✗ VLM调用失败\n")
            print(f"错误类型: {error_type}")
            print(f"错误代码: {error_code}")
            print(f"错误信息: {error_msg}")
            return False
        
        # 检查正常响应
        if response and hasattr(response, 'choices') and response.choices:
            if response.choices[0] and hasattr(response.choices[0], 'message'):
                answer = response.choices[0].message.content
                
                if answer and len(answer.strip()) > 0:
                    print(f"✅ VLM调用成功")
                    print(f"响应: {answer[:80]}")
                    return True
                else:
                    print(f"✗ VLM调用失败\n")
                    print(f"错误原因: API返回空内容")
                    return False
            else:
                print(f"✗ VLM调用失败\n")
                print(f"错误原因: 响应结构不完整")
                return False
        else:
            print(f"✗ VLM调用失败\n")
            print(f"错误原因: 响应格式异常")
            return False
            
    except Exception as e:
        print(f"✗ VLM调用失败\n")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)[:200]}")
        return False


if __name__ == "__main__":
    test_vlm_connection()
    print("\n" + "="*70)