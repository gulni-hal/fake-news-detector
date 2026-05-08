import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [newsText, setNewsText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async () => {
    if (!newsText) return alert("Lütfen bir haber metni girin!");
    
    setLoading(true);
    try {
      // FastAPI sunucumuza POST isteği atıyoruz
      const response = await axios.post('http://127.0.0.1:8000/predict', {
        text: newsText
      });
      setResult(response.data.prediction);
    } catch (error) {
      console.error("Hata:", error);
      alert("API'ye bağlanırken bir hata oluştu.");
    }
    setLoading(false);
  }

  return (
    <div className="container">
      <h1>Sahte Haber Tespit Sistemi</h1>
      <p>Şüphelendiğiniz İngilizce haber metnini aşağıya yapıştırın.</p>
      
      <textarea 
        rows="8" 
        cols="50"
        value={newsText}
        onChange={(e) => setNewsText(e.target.value)}
        placeholder="Haber metnini buraya yapıştırın..."
        style={{ width: '100%', padding: '10px', fontSize: '16px' }}
      />
      
      <br />
      <button 
        onClick={handlePredict} 
        disabled={loading}
        style={{ marginTop: '10px', padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
      >
        {loading ? 'Analiz Ediliyor...' : 'Analiz Et'}
      </button>

      {result && (
        <div style={{ 
          marginTop: '20px', 
          padding: '20px', 
          borderRadius: '8px',
          backgroundColor: result === 'Gerçek Haber' ? '#d4edda' : '#f8d7da',
          color: result === 'Gerçek Haber' ? '#155724' : '#721c24',
          fontWeight: 'bold',
          fontSize: '20px'
        }}>
          Sonuç: {result}
        </div>
      )}
    </div>
  )
}

export default App