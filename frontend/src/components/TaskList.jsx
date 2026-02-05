import './TaskList.css';

const TaskList = ({ tasks, onEdit, onDelete }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'No due date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getStatusClass = (status) => {
    const statusMap = {
      todo: 'status-todo',
      in_progress: 'status-progress',
      completed: 'status-completed',
    };
    return statusMap[status] || '';
  };

  const getPriorityClass = (priority) => {
    const priorityMap = {
      low: 'priority-low',
      medium: 'priority-medium',
      high: 'priority-high',
    };
    return priorityMap[priority] || '';
  };

  const getStatusLabel = (status) => {
    const labels = {
      todo: 'To Do',
      in_progress: 'In Progress',
      completed: 'Completed',
    };
    return labels[status] || status;
  };

  if (tasks.length === 0) {
    return (
      <div className="empty-state">
        <p>No tasks found. Create your first task!</p>
      </div>
    );
  }

  return (
    <div className="task-list">
      {tasks.map((task) => (
        <div key={task.id} className="task-card">
          <div className="task-header">
            <h3>{task.title}</h3>
            <div className="task-badges">
              <span className={`badge ${getStatusClass(task.status)}`}>
                {getStatusLabel(task.status)}
              </span>
              <span className={`badge ${getPriorityClass(task.priority)}`}>
                {task.priority.toUpperCase()}
              </span>
            </div>
          </div>

          {task.description && (
            <p className="task-description">{task.description}</p>
          )}

          <div className="task-footer">
            <div className="task-meta">
              <span className="task-date">📅 {formatDate(task.due_date)}</span>
              <span className="task-created">
                Created: {formatDate(task.created_at)}
              </span>
            </div>

            <div className="task-actions">
              <button
                onClick={() => onEdit(task)}
                className="btn-edit"
                title="Edit task"
              >
                ✏️ Edit
              </button>
              <button
                onClick={() => onDelete(task.id)}
                className="btn-delete"
                title="Delete task"
              >
                🗑️ Delete
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default TaskList;