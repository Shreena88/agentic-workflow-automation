import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  id?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = '', id }) => {
  return (
    <div id={id} className={`glass-card fade-in ${className}`}>
      {children}
    </div>
  );
};
