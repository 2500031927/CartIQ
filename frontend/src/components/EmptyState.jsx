import { Inbox } from "lucide-react";

export default function EmptyState({ icon: Icon = Inbox, title, description }) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Icon size={26} />
      </div>
      <h4>{title}</h4>
      {description && <p>{description}</p>}
    </div>
  );
}
