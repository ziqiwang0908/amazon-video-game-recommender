import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Video Game Recommender',
  description: 'Amazon review based video game recommendation demo'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
