// src/SelectPage.js
import React, { useState, useEffect } from 'react';
import { Layout, Typography, Cascader, Button, Row, Col, Card, Space, message, Spin } from 'antd';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ExperimentOutlined, SearchOutlined, PlayCircleOutlined, BarChartOutlined } from '@ant-design/icons';
import withPageTransition from './withPageTransition';
import axios from 'axios';
import config from './config';  // 添加这行

const { Content } = Layout;
const { Title, Paragraph } = Typography;

function SelectPage() {
    const [selectedPlant, setSelectedPlant] = useState([]);
    const [options, setOptions] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        fetchPlants();
    }, []);

    const fetchPlants = async () => {
        try {
            setLoading(true);
            const response = await axios.get(`${config.API_BASE_URL}/plants`);
            const data = response.data;
            const formattedOptions = Object.entries(data).map(([cropType, genoTypes]) => ({
                value: cropType,
                label: cropType,
                children: Object.entries(genoTypes).map(([genoType, plantIds]) => ({
                    value: genoType,
                    label: genoType,
                    children: plantIds.map(plantId => ({
                        value: plantId,
                        label: plantId,
                    })),
                })),
            }));
            setOptions(formattedOptions);
        } catch (error) {
            console.error('Error fetching plants:', error);
            message.error('Failed to load plant data. Please try again later.');
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (value) => {
        setSelectedPlant(value);
    };

    const handleAnalyze = async () => {
        if (selectedPlant.length === 3) {
            setLoading(true);
            try {
                const response = await axios.post(`${config.API_BASE_URL}/initialize-plant`, {
                    cropType: selectedPlant[0],
                    genoType: selectedPlant[1],
                    plantId: selectedPlant[2]
                });
                navigate('/home', { state: { plantData: response.data } });
            } catch (error) {
                console.error('Error initializing plant data:', error);
                message.error('Failed to initialize plant data. Please try again.');
            } finally {
                setLoading(false);
            }
        }
    };


    return (
        <Layout style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)' }}>
            <Content style={{ padding: '50px 50px' }}>
                <Row justify="center" gutter={[24, 24]}>
                    <Col xs={24} sm={22} md={20} lg={18} xl={16}>
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <Card style={{ borderRadius: '15px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)' }}>
                                <Title level={2} style={{ textAlign: 'center', marginBottom: 30, color: '#1890ff' }}>
                                    <ExperimentOutlined /> Plant Analysis System
                                </Title>
                                <Paragraph style={{ textAlign: 'center', fontSize: 16, marginBottom: 30 }}>
                                    Welcome to the Plant Analysis System. This system provides advanced image processing 
                                    and machine learning technologies to help you deeply analyze plant growth conditions, 
                                    optimize planting strategies, and improve crop yields.
                                </Paragraph>
                                <Row justify="center">
                                    <Col xs={24} sm={20} md={18} lg={16} xl={14}>
                                        <Space direction="vertical" size="large" style={{ width: '100%', marginBottom: 30 }}>
                                            <Spin spinning={loading}>
                                                <Cascader
                                                    options={options}
                                                    onChange={handleChange}
                                                    placeholder="Please select crop type, genotype, and specific plant"
                                                    style={{ width: '100%' }}
                                                    size="large"
                                                    expandTrigger="hover"
                                                />
                                            </Spin>
                                            <Button
                                                type="primary"
                                                icon={<SearchOutlined />}
                                                onClick={handleAnalyze}
                                                disabled={selectedPlant.length !== 3}
                                                loading={loading}
                                                block
                                                size="large"
                                                style={{ background: '#1890ff', borderColor: '#1890ff', color: '#fff' }}
                                            >
                                                Start Analysis
                                            </Button>
                                        </Space>
                                    </Col>
                                </Row>
                                <Row gutter={[16, 16]} justify="center">
                                    <Col xs={24} sm={12}>
                                        <Card
                                            hoverable
                                            style={{ borderRadius: '10px', background: '#e6f7ff' }}
                                        >
                                            <Space align="center">
                                                <PlayCircleOutlined style={{ fontSize: 32, color: '#1890ff', padding: '0 8px 0 0' }} />
                                                <div>
                                                    <Title level={4} style={{ margin: 0, color: '#1890ff' }}>Quick Start</Title>
                                                    <Paragraph style={{ margin: 0 }}>Watch tutorial videos to learn how to use the system</Paragraph>
                                                </div>
                                            </Space>
                                        </Card>
                                    </Col>
                                    <Col xs={24} sm={12}>
                                        <Card
                                            hoverable
                                            style={{ borderRadius: '10px', background: '#f9f0ff', padding: '0 8px 0 0' }}
                                        >
                                            <Space align="center">
                                                <BarChartOutlined style={{ fontSize: 32, color: '#722ed1' }} />
                                                <div>
                                                    <Title level={4} style={{ margin: 0, color: '#722ed1' }}>Analysis Reports</Title>
                                                    <Paragraph style={{ margin: 0 }}>View historical analysis reports and track growth trends</Paragraph>
                                                </div>
                                            </Space>
                                        </Card>
                                    </Col>
                                </Row>
                            </Card>
                        </motion.div>
                    </Col>
                </Row>
            </Content>
        </Layout>
    );
}


export default withPageTransition(SelectPage);