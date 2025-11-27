import React, { useEffect, useState } from 'react';
import { chatbotAPI } from '../services/api';

const ProductCard = ({ product, enableLiveCheck = false }) => {
  const {
    name,
    title,
    price_text,
    price_numeric,
    rating,
    url,
    image_url,
    description,
    similarity_score
  } = product;

  const [livePrice, setLivePrice] = useState(null);
  const [liveStatus, setLiveStatus] = useState(null);
  const [checkingLive, setCheckingLive] = useState(false);
  const [hasCheckedLive, setHasCheckedLive] = useState(false);

  const handleProductClick = () => {
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const handleLiveCheckClick = async (e) => {
    // Prevent triggering the main product click (which opens a new tab)
    e.stopPropagation();

    if (!enableLiveCheck) return;
    if (!product?.id && !url) return;
    if (checkingLive) return;

    try {
      setCheckingLive(true);
      const response = await chatbotAPI.checkLiveProduct(product?.id, url);
      if (response?.success) {
        if (response.latest_price_text) {
          setLivePrice(response.latest_price_text);
        }
        if (response.status) {
          setLiveStatus(response.status);
        }
        setHasCheckedLive(true);
      }
    } catch (err) {
      // Fail silently; fallback to scraped data
    } finally {
      setCheckingLive(false);
    }
  };

  const formatPrice = () => {
    if (livePrice) return livePrice;
    if (price_text) return price_text;
    if (price_numeric) return `KSh ${price_numeric.toLocaleString()}`;
    return 'Price not available';
  };

  const formatRating = (rating) => {
    if (!rating) return null;
    
    // Extract numeric rating if it's in format "4.2 out of 5" or "4 out of 5"
    const match = rating.toString().match(/(\d+\.?\d*)/);
    const numericRating = match ? parseFloat(match[1]) : null;
    
    if (numericRating) {
      const stars = Math.round(numericRating);
      return (
        <div className="flex items-center space-x-1">
          <div className="flex">
            {[...Array(5)].map((_, i) => (
              <svg
                key={i}
                className={`w-4 h-4 ${i < stars ? 'text-yellow-400' : 'text-gray-300'}`}
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            ))}
          </div>
          <span className="text-sm text-gray-600">({numericRating})</span>
        </div>
      );
    }
    
    return <span className="text-sm text-gray-600">{rating}</span>;
  };

  return (
    <div 
      className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow duration-200 cursor-pointer"
      onClick={handleProductClick}
    >
      {/* Product Image */}
      <div className="relative h-48 bg-gray-100">
        {image_url ? (
          <img
            src={image_url}
            alt={name || title}
            className="w-full h-full object-contain p-2"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        
        {/* Fallback when image fails to load */}
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-400" style={{ display: image_url ? 'none' : 'flex' }}>
          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        
        {/* Similarity Score Badge */}
        {similarity_score && (
          <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
            {Math.round(similarity_score * 100)}% match
          </div>
        )}
      </div>

      {/* Product Details */}
      <div className="p-4">
        {/* Product Name */}
        <h3 className="font-semibold text-gray-800 text-sm mb-2 line-clamp-2 leading-tight">
          {name || title || 'Product Name'}
        </h3>

        {/* Price */}
        <div className="mb-2">
          <span className="text-lg font-bold text-green-600">
            {formatPrice()}
          </span>
          {liveStatus && (
            <span
              className={`ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                liveStatus === 'in_stock'
                  ? 'bg-green-100 text-green-700'
                  : liveStatus === 'out_of_stock'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-gray-100 text-gray-700'
              }`}
            >
              {liveStatus === 'in_stock' && 'In stock (live)'}
              {liveStatus === 'out_of_stock' && 'Out of stock (live)'}
              {liveStatus !== 'in_stock' && liveStatus !== 'out_of_stock' && 'Status unknown'}
            </span>
          )}
        </div>

        {/* Rating */}
        {rating && (
          <div className="mb-2">
            {formatRating(rating)}
          </div>
        )}

        {/* Description */}
        {description && (
          <p className="text-gray-600 text-xs mb-3 line-clamp-2">
            {description}
          </p>
        )}

        {/* Action Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            handleProductClick();
          }}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white text-sm py-2 px-4 rounded-md transition-colors duration-200 flex items-center justify-center space-x-2 mb-2"
        >
          <span>View Product</span>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </button>

        {enableLiveCheck && (
          <button
            onClick={handleLiveCheckClick}
            disabled={checkingLive}
            className="w-full border border-blue-500 text-blue-500 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed text-xs py-1.5 px-3 rounded-md transition-colors duration-200"
          >
            {checkingLive
              ? 'Checking live price...'
              : hasCheckedLive
              ? 'Live price checked'
              : 'Check live price'}
          </button>
        )}
      </div>
    </div>
  );
};

export default ProductCard;
