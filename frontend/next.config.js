/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  poweredByHeader: false,
  async rewrites() {
    const backend = process.env.AIWO_RAG_BASE_URL;
    if (!backend) return [];
    return [
      {
        source: "/aiwo-images/:path*",
        destination: `${backend.replace(/\/+$/, "")}/images/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "Permissions-Policy",
            value: "microphone=self, camera=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
