import { CircleAlert } from 'lucide-react';
export function ErrorMessage({ message }: { message: string }) {
  return <div className="error-message" role="alert"><CircleAlert size={19} /><div><strong>Chưa thể hoàn tất phân tích</strong><p>{message}</p></div></div>;
}
