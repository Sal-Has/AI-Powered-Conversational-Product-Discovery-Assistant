@echo off
echo Cleaning up unnecessary scraper files...

cd backend

echo Removing vector database files (not working)...
del enhanced_jumia_scraper.py
del vector_scraper.py
del simple_enhanced_scraper.py

echo Removing test files...
del test_enhanced_scraper.py
del test_pipeline.py
del quick_vector_test.py

echo Removing original scraper...
del your_enhanced_scraper.py

echo Removing product pipeline (replaced by basic scraper)...
del product_pipeline.py

cd ..

echo Removing vector requirements file...
del requirements_vector.txt

echo Removing vector environment folder...
rmdir /s /q vector_env

echo ✅ Cleanup complete!
echo.
echo 📁 Remaining files:
echo   - backend/basic_scraper_with_storage.py (WORKING SCRAPER)
echo   - backend/jumia_products.db (YOUR DATA)
echo   - backend/jumia_smartphones_basic.xlsx (EXCEL EXPORT)
echo   - backend/app.py, auth.py, models.py, config.py (FLASK APP)
echo   - requirements.txt (BASIC DEPENDENCIES)
echo.
echo 🚀 To run scraper: cd backend && python basic_scraper_with_storage.py
pause
