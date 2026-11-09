'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { setAccessToken } from '@/lib/api';

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [mounted, setMounted] = useState(false);
  const { data: session } = useSession();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    // Only set the access token on the client side
    if (mounted) {
      if (session?.accessToken) {
        setAccessToken(session.accessToken as string);
      } else {
        setAccessToken(null);
      }
    }
  }, [session, mounted]);

  return <>{children}</>;
}
