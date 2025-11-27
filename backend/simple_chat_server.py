    # Import and register chat blueprint
    try:
        from chat_api_no_auth import chat_no_auth_bp
        app.register_blueprint(chat_no_auth_bp, url_prefix='/api')
        print("✅ Chat API registered successfully")
    except Exception as e:
        print(f"❌ Failed to register chat API: {e}")
        return None
