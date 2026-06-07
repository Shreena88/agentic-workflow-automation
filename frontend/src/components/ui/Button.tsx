import React, { type ButtonHTMLAttributes } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'success' | 'danger-outline' | 'default';
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'default', 
  icon,
  className = '',
  ...props 
}) => {
  const baseClass = 'btn';
  const variantClass = variant !== 'default' ? `btn-${variant}` : '';

  return (
    <button className={`${baseClass} ${variantClass} ${className}`} {...props}>
      {icon && <span className="btn-icon">{icon}</span>}
      {children}
    </button>
  );
};
