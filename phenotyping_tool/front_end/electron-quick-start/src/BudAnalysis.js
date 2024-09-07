import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { Layout, Row, Col, Card, Button, Space, Spin, message, Tabs } from 'antd';
import { LeftOutlined, RightOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import axios from 'axios';
import config from './config';
import withPageTransition from "./withPageTransition";

const { Content } = Layout;
const { TabPane } = Tabs;

const ORIGINAL_SIZE = 640; // 原始图片大小
const DISPLAY_SIZE = 298; // 调整显示大小为298

const BudAnalysis = () => {
  const { plantId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { date, images: initialImages } = location.state || {};

  const [loading, setLoading] = useState(true);
  const [budData, setBudData] = useState(null);
  const [currentView, setCurrentView] = useState('sv-000');
  const [selectedBud, setSelectedBud] = useState(null);
  const [images, setImages] = useState(initialImages || {});

  const views = ['sv-000', 'sv-045', 'sv-090'];

  useEffect(() => {
    fetchBudData();
    if (!initialImages) {
      fetchImages();
    }
  }, [plantId, date]);

  useEffect(() => {
    if (budData) {
      console.log("Full bud data:", budData);
    }
  }, [budData]);

  const fetchBudData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${config.API_BASE_URL}/bud-analysis/${plantId}/${date}`);
      setBudData(response.data);
    } catch (error) {
      console.error('Error fetching bud data:', error);
      message.error('Failed to fetch bud data');
    } finally {
      setLoading(false);
    }
  };

  const fetchImages = async () => {
    try {
      const response = await axios.get(`${config.API_BASE_URL}/plant-images/${plantId}/${date}`);
      setImages(response.data);
    } catch (error) {
      console.error('Error fetching images:', error);
      message.error('Failed to fetch images');
    }
  };

  const changeView = (direction) => {
    const currentIndex = views.indexOf(currentView);
    let newIndex;
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % views.length;
    } else {
      newIndex = (currentIndex - 1 + views.length) % views.length;
    }
    setCurrentView(views[newIndex]);
    setSelectedBud(null);
  };

  const renderBudBoxes = () => {
    if (!budData || !budData[currentView]) {
      console.log("No bud data for current view:", currentView);
      return null;
    }

    console.log("Bud data for current view:", budData[currentView]);

    const scale = DISPLAY_SIZE / ORIGINAL_SIZE;
    return budData[currentView].map((bud, index) => {
      console.log("Processing bud:", bud);
      const [x, y, width, height] = Array.isArray(bud.boxes[0]) ? bud.boxes[0] : bud.boxes;
      console.log("Bud coordinates:", { x, y, width, height });
      const isSelected = selectedBud === index;
      
      // 调整坐标计算
      const scaledX = x * scale;
      const scaledY = y * scale;
      const scaledWidth = (width - x) * scale;
      const scaledHeight = (height - y) * scale;

      return (
        <rect
          key={index}
          x={scaledX}
          y={scaledY}
          width={Math.max(scaledWidth, 5)} // 最小宽度为5像素
          height={Math.max(scaledHeight, 5)} // 最小高度为5像素
          stroke={isSelected ? 'red' : 'green'}
          strokeWidth="1"
          fill="none"
        />
      );
    });
  };

  const handleBack = () => {
    navigate(-1); // 返回上一页
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content style={{ padding: '24px' }}>
        <Card 
          title={`Bud Analysis: ${plantId}`} 
          style={{ marginBottom: 24 }}
          extra={
            <Button icon={<ArrowLeftOutlined />} onClick={handleBack}>
              Back
            </Button>
          }
        >
          <Row gutter={24}>
            <Col span={14}>
              <Card style={{ marginBottom: 16 }}>
                <div style={{ width: DISPLAY_SIZE, height: DISPLAY_SIZE, margin: '0 auto', position: 'relative' }}>
                  {images[currentView] && (
                    <>
                      <img
                        src={`${config.API_BASE_URL}${images[currentView]}`}
                        alt="Plant"
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover'
                        }}
                      />
                      <svg 
                        style={{ 
                          position: 'absolute', 
                          top: 0, 
                          left: 0, 
                          width: '100%', 
                          height: '100%' 
                        }}
                        viewBox={`0 0 ${DISPLAY_SIZE} ${DISPLAY_SIZE}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {renderBudBoxes()}
                      </svg>
                    </>
                  )}
                </div>
              </Card>
              <Space style={{ width: '100%', justifyContent: 'center' }}>
                <Button icon={<LeftOutlined />} onClick={() => changeView('prev')}>
                  Previous View
                </Button>
                <Button icon={<RightOutlined />} onClick={() => changeView('next')}>
                  Next View
                </Button>
              </Space>
            </Col>
            <Col span={10}>
              <Spin spinning={loading}>
                <Tabs defaultActiveKey="1">
                  <TabPane tab="Bud Summary" key="1">
                    <Card>
                      <p>Total Buds: {budData && budData[currentView] ? budData[currentView].length : 0}</p>
                      {/* Add more summary information here */}
                    </Card>
                  </TabPane>
                  <TabPane tab="Bud Details" key="2">
                    <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                      {budData && budData[currentView] && budData[currentView].map((bud, index) => (
                        <Card
                          key={index}
                          style={{ 
                            marginBottom: 16, 
                            cursor: 'pointer',
                            borderColor: selectedBud === index ? 'red' : undefined
                          }}
                          onClick={() => setSelectedBud(index)}
                        >
                          <p>Bud {index + 1}</p>
                          <p>Coordinates: {Array.isArray(bud.boxes[0]) ? bud.boxes[0].join(', ') : bud.boxes.join(', ')}</p>
                          <p>Confidence: {Array.isArray(bud.conf) 
                            ? bud.conf[0].toFixed(4) 
                            : (typeof bud.conf === 'number' ? bud.conf.toFixed(4) : 'N/A')}
                          </p>
                        </Card>
                      ))}
                    </div>
                  </TabPane>
                </Tabs>
              </Spin>
            </Col>
          </Row>
        </Card>
      </Content>
    </Layout>
  );
};

export default withPageTransition(BudAnalysis);