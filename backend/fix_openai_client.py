        response = client.chat.completions.create(
            model="gpt-3.5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, OpenAI is working!'"}
            ],
            max_tokens=50
        )
{{ ... }}
        # Test API call
        response = client.chat.completions.create(
            model="gpt-3.5",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": "Recommend a good smartphone under $500"}
            ],
            max_tokens=100
        )
