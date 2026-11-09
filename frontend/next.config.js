const createNextIntlPlugin = require('next-intl/plugin');

/** @type {import('next').NextConfig} */
const nextConfig = {
  // A separate local build directory lets the current production server keep serving.
  distDir: process.env.WARDROWBE_BUILD_DIR || '.next',
  output: 'standalone',
  experimental: {
    instrumentationHook: true,
    // Disable automatic static optimization for pages using client-side context
    missingSuspenseWithCSRBailout: false,
  },
  images: {
    unoptimized: true,
  },
  // Skip type checking and linting during build (already done in CI)
  typescript: {
    ignoreBuildErrors: true,
    // Staged local builds keep Next's generated type includes out of the live tsconfig.
    tsconfigPath: process.env.WARDROWBE_BUILD_TSCONFIG || 'tsconfig.json',
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  // /api/v1/* is proxied by app/api/v1/[...path]/route.ts rather than a rewrite here, because
  // rewrites() is serialized into routes-manifest.json at build time and so cannot honor a
  // runtime BACKEND_URL in the prebuilt image.
};

const withNextIntl = createNextIntlPlugin();
module.exports = withNextIntl(nextConfig);
