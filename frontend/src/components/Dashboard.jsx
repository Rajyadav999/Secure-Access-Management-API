import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { taskAPI } from '../services/api';
import TaskForm from './TaskForm';
import TaskList from './TaskList';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  
  const [tasks, setTasks] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [filters, setFilters] = useState({
    status: '',
    priority: '',
    search: '',
  });
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 10,
  });

  useEffect(() => {
    fetchTasks();
    fetchStats();
  }, [filters, pagination]);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      const params = {
        ...pagination,
        ...filters,
      };
      
      // Remove empty filters
      Object.keys(params).forEach(key => {
        if (params[key] === '') delete params[key];
      });
      
      const response = await taskAPI.getTasks(params);
      setTasks(response.data.tasks);
    } catch (error) {
      console.error('Error fetching tasks:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await taskAPI.getStats();
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleCreateTask = async (taskData) => {
    try {
      await taskAPI.createTask(taskData);
      setShowForm(false);
      fetchTasks();
      fetchStats();
    } catch (error) {
      console.error('Error creating task:', error);
      throw error;
    }
  };

  const handleUpdateTask = async (taskData) => {
    try {
      await taskAPI.updateTask(editingTask.id, taskData);
      setEditingTask(null);
      setShowForm(false);
      fetchTasks();
      fetchStats();
    } catch (error) {
      console.error('Error updating task:', error);
      throw error;
    }
  };

  const handleDeleteTask = async (taskId) => {
    if (window.confirm('Are you sure you want to delete this task?')) {
      try {
        await taskAPI.deleteTask(taskId);
        fetchTasks();
        fetchStats();
      } catch (error) {
        console.error('Error deleting task:', error);
      }
    }
  };

  const handleEdit = (task) => {
    setEditingTask(task);
    setShowForm(true);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <h1>Task Manager</h1>
          <div className="user-info">
            <span>Welcome, {user?.full_name || user?.username}!</span>
            {user?.role === 'admin' && <span className="badge">Admin</span>}
            <button onClick={handleLogout} className="btn-secondary">
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="dashboard-main">
        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <h3>Total Tasks</h3>
              <p className="stat-value">{stats.total_tasks}</p>
            </div>
            <div className="stat-card">
              <h3>To Do</h3>
              <p className="stat-value">{stats.by_status.todo}</p>
            </div>
            <div className="stat-card">
              <h3>In Progress</h3>
              <p className="stat-value">{stats.by_status.in_progress}</p>
            </div>
            <div className="stat-card">
              <h3>Completed</h3>
              <p className="stat-value">{stats.by_status.completed}</p>
              <p className="stat-subtitle">{stats.completed_percentage}%</p>
            </div>
          </div>
        )}

        <div className="tasks-section">
          <div className="section-header">
            <h2>My Tasks</h2>
            <button
              onClick={() => {
                setEditingTask(null);
                setShowForm(!showForm);
              }}
              className="btn-primary"
            >
              {showForm ? 'Cancel' : '+ New Task'}
            </button>
          </div>

          {showForm && (
            <TaskForm
              task={editingTask}
              onSubmit={editingTask ? handleUpdateTask : handleCreateTask}
              onCancel={() => {
                setShowForm(false);
                setEditingTask(null);
              }}
            />
          )}

          <div className="filters">
            <input
              type="text"
              placeholder="Search tasks..."
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
              className="search-input"
            />
            
            <select
              value={filters.status}
              onChange={(e) => setFilters({ ...filters, status: e.target.value })}
            >
              <option value="">All Status</option>
              <option value="todo">To Do</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
            </select>

            <select
              value={filters.priority}
              onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
            >
              <option value="">All Priority</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          {loading ? (
            <div className="loading">Loading tasks...</div>
          ) : (
            <TaskList
              tasks={tasks}
              onEdit={handleEdit}
              onDelete={handleDeleteTask}
            />
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;