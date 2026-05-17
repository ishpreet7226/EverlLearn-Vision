import { Inter } from "next/font/google";
import "./globals.css";
import ThemeProvider from "./components/ThemeProvider";
import { ToastProvider } from "./components/Toast";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata = {
  title: "EverLearn ML — Build Your Own Image Classifier",
  description:
    "Teachable Machine for image classification. Create classes, upload images, train a model, and test predictions — all from the browser. Powered by PyTorch and FastAPI.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <ThemeProvider>
          <ToastProvider>{children}</ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
