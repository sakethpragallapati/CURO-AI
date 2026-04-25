/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['reagraph', 'three', '@react-three/fiber', '@react-three/drei'],
  reactStrictMode: false, 
  swcMinify: false,
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.output.globalObject = 'self';
    }
    return config;
  }
}

module.exports = nextConfig