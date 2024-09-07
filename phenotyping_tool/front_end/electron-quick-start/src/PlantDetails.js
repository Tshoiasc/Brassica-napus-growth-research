// src/PlantDetails.js
import React from 'react';
import { useParams, Link,useNavigate } from 'react-router-dom';
import { Layout, Card, Typography, Descriptions, Spin, Button } from 'antd';
import { useState, useEffect } from 'react';
import axios from 'axios';
import config from './config';
import withPageTransition from './withPageTransition';
const { Content } = Layout;
const { Title } = Typography;

function PlantDetails() {
  const [plantData, setPlantData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { plantId } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPlantDetails = async () => {
      try {
        const response = await axios.get(`${config.API_BASE_URL}/plant-details/${plantId}`);
        setPlantData(response.data);
      } catch (error) {
        console.error('Error fetching plant details:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPlantDetails();
  }, [plantId]);

  if (loading) {
    return <Spin size="large" />;
  }

  if (!plantData) {
    return <div>No data available</div>;
  }

  const handleGoBack = () => {
    navigate(-1); // 这将返回上一页
  };
  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content style={{ padding: '24px' }}>
        <Card>
          <Title level={2}>Plant Details - {plantId}</Title>
          <Descriptions bordered column={2}>
            <Descriptions.Item label="Genotype Name">{plantData.genotype_name || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Genotype ID">{plantData.genotype_id || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Location">{plantData.location || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Treatment">{plantData.treatment || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Repetition">{plantData.rep || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Raceme Start (DAS)">{plantData.R01_raceme_start || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Mean Extension Rate">{plantData.R02_mean_rate || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Max Extension Rate">{plantData.R03_max_rate || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Max Rate Time (DAS)">{plantData.R04_DAS_of_max_rate || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Main Extension Period">{plantData.R05_main_ext_period || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="50% Height Period">{plantData.R06_50ht_period || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="75% Height Period">{plantData.R07_75ht_period || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="50% Height Rate">{plantData.R08_50ht_rate || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="75% Height Rate">{plantData.R09_75ht_rate || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Max Height">{plantData.R10_max_height || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Max Height Time (DAS)">{plantData.R11_max_height_DAS || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Days to Max Height">{plantData.R12_extension_days_to_max || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Raceme Collapse">{plantData.R13_raceme_collapse || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="BBCH51 to Raceme Start">{plantData.R14_BBCH51_to_raceme_start || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Elongation in Vernalization">{plantData.R15_elong_in_vern || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="First Flower (DAS)">{plantData.F01_first_flowers_DAS || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Peak Flowering (DAS)">{plantData.F02_peak_flowers_DAS || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Days to Peak Flowering">{plantData.F03_flowering_days_to_peak || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Peak Pixel Day">{plantData.F06_pixel?.day_peak || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Pixel Days to Peak">{plantData.F07_pixel?.days_to_peak || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Mean Pixel Days to Peak">{plantData.F09_mean_pixel?.days_to_peak || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Ground BBCH51">{plantData.Ground_BBCH51 || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Ground BBCH60">{plantData.Ground_BBCH60 || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Ground 60-51">{plantData['Ground_60-51'] || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Auto60-ground51">{plantData['Auto60-ground51'] || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Total Flowering Days">{plantData.F04_total_flowering_days || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Total Pixel Days">{plantData.F08_total_pixel?.days || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Flowering Finished">{plantData.F05_flowering_finished || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Date of Collapse">{plantData['date of collapse'] || 'N/A'}</Descriptions.Item>
          </Descriptions>
          <Button type="primary" onClick={handleGoBack} style={{ marginTop: '20px' }}>
            Back
          </Button>
        </Card>
      </Content>
    </Layout>
  );
}

export default withPageTransition(PlantDetails);