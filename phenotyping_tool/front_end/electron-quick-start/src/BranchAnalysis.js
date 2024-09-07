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
const DISPLAY_SIZE = 298; // 显示大小

const BranchAnalysis = () => {
  const { plantId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { date, images: initialImages } = location.state || {};

  const [loading, setLoading] = useState(true);
  const [branchData, setBranchData] = useState(null);
  const [currentView, setCurrentView] = useState('sv-000');
  const [selectedBranch, setSelectedBranch] = useState(null);
  const [images, setImages] = useState(initialImages || {});
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const containerRef = useRef(null);
  const views = ['sv-000', 'sv-045', 'sv-090'];

  useEffect(() => {
    console.log("Component mounted or updated");
    fetchBranchData();
    if (!initialImages) {
      fetchImages();
    }
  }, [plantId, currentView, date]);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect();
        setContainerSize({ width, height: width });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  const fetchBranchData = async () => {
    setLoading(true);
    try {
      console.log(`Fetching branch data for ${plantId}, ${date}, ${currentView}`);
      const response = await axios.get(`${config.API_BASE_URL}/branch-analysis/${plantId}/${date}/${currentView}`);
      console.log("Branch data received:", response.data);
      setBranchData(response.data.branches);
    } catch (error) {
      console.error('Error fetching branch data:', error);
      message.error('Failed to fetch branch data');
    } finally {
      setLoading(false);
    }
  };

  const fetchImages = async () => {
    try {
      console.log(`Fetching images for ${plantId}, ${date}`);
      const response = await axios.get(`${config.API_BASE_URL}/plant-images/${plantId}/${date}`);
      console.log("Images received:", response.data);
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
    console.log(`Changing view from ${currentView} to ${views[newIndex]}`);
    setCurrentView(views[newIndex]);
    setSelectedBranch(null);
  };

  const renderBranches = () => {
    if (!branchData) {
      console.log("No branch data available");
      return null;
    }
    console.log("Rendering branches:", branchData.length);
    return branchData.map((branch, index) => (
      <polyline
        key={index}
        points={branch.branch_path.map(point => point.join(',')).join(' ')}
        stroke={selectedBranch === index ? 'red' : 'yellow'}
        strokeWidth="2"
        fill="none"
      />
    ));
  };

  const renderBranchDetails = (branch, index) => (
    <Card 
      key={index} 
      style={{ 
        marginBottom: 16, 
        cursor: 'pointer',
        borderColor: selectedBranch === index ? 'red' : undefined
      }}
      onClick={() => setSelectedBranch(index)}
    >
      <p><strong>Branch {index + 1}</strong></p>
      <p>Level: {branch.level}</p>
      <p>Length: {branch.length.toFixed(2)}</p>
      <p>Vertical Length: {branch.vertical_length.toFixed(2)}</p>
      <p>Horizontal Length: {branch.horizontal_length.toFixed(2)}</p>
      {branch.angle !== null && <p>Angle: {branch.angle.toFixed(2)}°</p>}
    </Card>
  );

  const handleBack = () => {
    navigate(-1); // 返回上一页
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content style={{ padding: '24px' }}>
        <Card 
          title={`Plant Analysis: ${plantId}`} 
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
                        viewBox={`0 0 ${ORIGINAL_SIZE} ${ORIGINAL_SIZE}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {renderBranches()}
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
                  <TabPane tab="Branch Summary" key="1">
                    <Card>
                      <p>Total Branches: {branchData ? branchData.length : 0}</p>
                      <p>Main Stem Length: {branchData && branchData[0] ? branchData[0].length.toFixed(2) : 'N/A'}</p>
                      {/* Add more summary information here */}
                    </Card>
                  </TabPane>
                  <TabPane tab="Branch Details" key="2">
                    <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                      {branchData && branchData.map((branch, index) => renderBranchDetails(branch, index))}
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

export default withPageTransition(BranchAnalysis);