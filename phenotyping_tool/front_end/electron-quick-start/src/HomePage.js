// src/HomePage.js
import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Layout,
  Row,
  Col,
  Card,
  Image,
  Typography,
  Space,
  Button,
  Menu,
  List,
  Statistic,
  Breadcrumb,
  Drawer,
  Skeleton,
  Divider,
  message
} from "antd";
import { Link, useNavigate, useLocation } from "react-router-dom";
import {
  CalendarOutlined,
  ExperimentOutlined,
  BranchesOutlined,
  MenuOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  LeftOutlined,
  RightOutlined,
} from "@ant-design/icons";
import { motion, AnimatePresence } from "framer-motion";
import withPageTransition from "./withPageTransition";
import axios from "axios";
import config from "./config";
import { Image as AntImage } from "antd"; // 重命名 Image 组件以避免与 HTML Image 元素冲突
import fileDownload from 'js-file-download';
const { Content } = Layout;
const { Text, Title } = Typography;

const viewAngles = ["sv-000", "sv-045", "sv-090", "tv-000"];
const viewAngleLabels = {
  "sv-000": "Side View 0°",
  "sv-045": "Side View 45°",
  "sv-090": "Side View 90°",
  "tv-000": "Top View",
};
// 添加这个常量来设置图片容器的宽高比
const IMAGE_ASPECT_RATIO = 4 / 3; // 可以根据实际图片比例调整

function HomePage() {
  const [selectedDate, setSelectedDate] = useState("");
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [currentViewAngle, setCurrentViewAngle] = useState("sv-000");
  const [loading, setLoading] = useState(true);
  const [plantInfo, setPlantInfo] = useState(null);
  const [budInfo, setBudInfo] = useState(null);
  const [branchInfo, setBranchInfo] = useState(null);
  const [images, setImages] = useState({});
  const [dateList, setDateList] = useState([]);
  const navigate = useNavigate();
  const [containerWidth, setContainerWidth] = useState(0);
  const location = useLocation();

  const { plantData } = location.state || {};
  console.log("Plant data received:", plantData);

  const [imageHeight, setImageHeight] = useState(0);
  const [initialImageSize, setInitialImageSize] = useState({
    width: 0,
    height: 0,
  });

  const [aspectRatio, setAspectRatio] = useState(4 / 3); // 默认宽高比
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const handleImageLoad = useCallback((event) => {
    const img = event.target;
    const newAspectRatio = img.naturalWidth / img.naturalHeight;
    setAspectRatio(newAspectRatio);
  }, []);

  useEffect(() => {
    if (containerRef.current) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          setContainerSize({ width, height });
        }
      });

      resizeObserver.observe(containerRef.current);

      return () => {
        resizeObserver.disconnect();
      };
    }
  }, []);

  useEffect(() => {
    if (plantData && plantData.dates && plantData.dates.length > 0) {
      setDateList(plantData.dates);
      setSelectedDate(plantData.dates[0]);
      fetchAllData(plantData.dates[0]);
    }
  }, [plantData]);

  const fetchAllData = async (date) => {
    if (!plantData || !plantData.plantId) {
      console.error("Plant data is missing");
      return;
    }
    setLoading(true);
    try {
      const [imagesRes, infoRes, budRes, branchRes] = await Promise.all([
        axios.get(
          `${config.API_BASE_URL}/plant-images/${plantData.plantId}/${date}`
        ),
        axios.get(`${config.API_BASE_URL}/plant-info/${plantData.plantId}`),
        axios.get(
          `${config.API_BASE_URL}/bud-info/${plantData.plantId}/${date}`
        ),
        axios.get(
          `${config.API_BASE_URL}/branch-info/${plantData.plantId}/${date}`
        ),
      ]);
      console.log("Images response:", imagesRes.data);
      console.log("Plant info response:", infoRes.data);
      console.log("Bud info response:", budRes.data);
      console.log("Branch info response:", branchRes.data);
      setImages(imagesRes.data);
      setPlantInfo(infoRes.data);
      setBudInfo(budRes.data);
      setBranchInfo(branchRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };
  const handleExportReport = async () => {
    try {
      const response = await axios.post(
        `${config.API_BASE_URL}/generate-report`,
        {
          plant_id: plantData.plantId,
          date: selectedDate,
        },
        {
          responseType: 'blob',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
  
      // 检查响应状态
      if (response.status === 200) {
        const contentDisposition = response.headers['content-disposition'];
        let filename = 'report.pdf';
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
          if (filenameMatch.length === 2)
            filename = filenameMatch[1];
        }
  
        fileDownload(response.data, filename);
        message.success('Report generated and downloaded successfully');
      } else {
        throw new Error('Failed to generate report');
      }
    } catch (error) {
      console.error('Error generating report:', error);
      message.error('Failed to generate report: ' + (error.response?.data?.detail || error.message));
    }
  };

  const showDrawer = () => {
    setDrawerVisible(true);
  };

  const onClose = () => {
    setDrawerVisible(false);
  };

  const changeViewAngle = (direction) => {
    const currentIndex = viewAngles.indexOf(currentViewAngle);
    let newIndex;
    if (direction === "next") {
      newIndex = (currentIndex + 1) % viewAngles.length;
    } else {
      newIndex = (currentIndex - 1 + viewAngles.length) % viewAngles.length;
    }
    setCurrentViewAngle(viewAngles[newIndex]);
  };

  const breadcrumbItems = [
    { title: <Link to="/">{plantData?.cropType || "Brassica Napus"}</Link> },
    { title: <Link to="/">{plantData?.genoType || "Genotype 1"}</Link> },
    { title: <Link to="/">{plantData?.plantId || "Plant 1"}</Link> },
  ];

  return (
    <Layout style={{ minHeight: "100vh", background: "#f0f2f5" }}>
      <Content style={{ padding: "24px", position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: 24,
            left: 24,
            right: 24,
            zIndex: 1,
          }}
        >
          <Row
            justify="space-between"
            align="middle"
            style={{ marginBottom: "16px" }}
          >
            <Col>
              <Breadcrumb
                items={breadcrumbItems}
                separator=">"
                style={{ fontSize: "14px" }}
              />
            </Col>
            <Col>
              <Space>
              <Button 
                  icon={<FilePdfOutlined />} 
                  style={{ fontSize: "13px" }}
                  onClick={handleExportReport}
                >
                  Export Report
                </Button>
                <Button
                  danger
                  icon={<BranchesOutlined />}
                  style={{
                    fontSize: "13px",
                    backgroundColor: "#fff1f0",
                    borderColor: "#ffa39e",
                  }}
                  onMouseEnter={(e) =>
                    (e.target.style.backgroundColor = "#ffccc7")
                  }
                  onMouseLeave={(e) =>
                    (e.target.style.backgroundColor = "#fff1f0")
                  }
                >
                  Bud Trajectory Information
                </Button>
                <Button
                  icon={<MenuOutlined />}
                  onClick={showDrawer}
                  style={{ fontSize: "13px" }}
                >
                  Select Date
                </Button>
              </Space>
            </Col>
          </Row>
        </div>
        <Drawer
          title={<span style={{ fontSize: "16px" }}>Select Date</span>}
          placement="left"
          onClose={onClose}
          open={drawerVisible}
        >
          <Menu
            mode="vertical"
            selectedKeys={[selectedDate]}
            onClick={({ key }) => {
              setSelectedDate(key);
              fetchAllData(key);
              onClose();
            }}
            style={{ fontSize: "14px" }}
          >
            {dateList.map((date) => (
              <Menu.Item key={date} icon={<CalendarOutlined />}>
                {date}
              </Menu.Item>
            ))}
          </Menu>
        </Drawer>
        <Row gutter={[24, 24]} style={{ marginTop: "48px" }}>
          <Col xs={24} md={10}>
            <Card
              hoverable
              style={{ height: "100%", position: "relative" }}
              bodyStyle={{ padding: 0, height: "100%" }}
            >
              <div
                ref={containerRef}
                style={{
                  position: "relative",
                  width: "100%",
                  paddingTop: `${(1 / aspectRatio) * 100}%`,
                  background: "#f0f2f5",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {/* 视角标签和切换按钮 */}
                  <div
                    style={{
                      position: "absolute",
                      top: 10,
                      left: 10,
                      zIndex: 2,
                      background: "rgba(0,0,0,0.5)",
                      color: "white",
                      padding: "5px 10px",
                      borderRadius: "4px",
                    }}
                  >
                    {viewAngleLabels[currentViewAngle]}
                  </div>
                  <Button
                    icon={<LeftOutlined />}
                    onClick={() => changeViewAngle("prev")}
                    style={{
                      position: "absolute",
                      left: 10,
                      top: "50%",
                      transform: "translateY(-50%)",
                      zIndex: 2,
                    }}
                  />
                  <Button
                    icon={<RightOutlined />}
                    onClick={() => changeViewAngle("next")}
                    style={{
                      position: "absolute",
                      right: 10,
                      top: "50%",
                      transform: "translateY(-50%)",
                      zIndex: 2,
                    }}
                  />
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={currentViewAngle}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {loading ? (
                        <Skeleton.Image
                          style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "contain",
                          }}
                          active
                        />
                      ) : images && images[currentViewAngle] ? (
                        <AntImage
                          src={`${config.API_BASE_URL}${images[currentViewAngle]}`}
                          alt="Plant Image"
                          style={{
                            maxWidth: "100%",
                            maxHeight: "100%",
                            objectFit: "contain",
                          }}
                          onLoad={handleImageLoad}
                          preview={{
                            mask: "Preview",
                            maskClassName: "custom-mask-class",
                          }}
                        />
                      ) : (
                        <div>No image available</div>
                      )}
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
              <div style={{ padding: "16px" }}>
                <Card.Meta
                  title={
                    <span
                      style={{
                        fontSize: "15px",
                        color: "#999",
                        fontWeight: "normal",
                      }}
                    >{`Capture Date: ${selectedDate}`}</span>
                  }
                />
                <Card.Meta
                  title={
                    <span
                      style={{
                        fontSize: "15px",
                        color: "#999",
                        fontWeight: "normal",
                        marginTop: "8px",
                      }}
                    >{`Note: ${selectedDate}`}</span>
                  }
                />
              </div>
            </Card>
          </Col>
          <Col xs={24} md={14}>
            <Space direction="vertical" size="large" style={{ width: "100%" }}>
              <Card
                title={
                  <span style={{ fontSize: "16px" }}>
                    Plant Basic Information
                  </span>
                }
                extra={
                  <Link
                    to={`/plant-details/${plantData?.plantId}`}
                    style={{ fontSize: "13px" }}
                  >
                    Details
                  </Link>
                }
              >
                <Skeleton loading={loading} active>
                  {plantInfo && (
                    <List
                      size="small"
                      dataSource={[
                        { title: "CropType", value: plantData?.cropType },
                        { title: "GenoType", value: plantData?.genoType },
                      ]}
                      renderItem={(item) => (
                        <List.Item>
                          <Text strong style={{ fontSize: "14px" }}>
                            {item.title}:
                          </Text>{" "}
                          <Text style={{ fontSize: "14px" }}>{item.value}</Text>
                        </List.Item>
                      )}
                    />
                  )}
                </Skeleton>
              </Card>
              <Card
                title={<span style={{ fontSize: "16px" }}>Bud Analysis</span>}
                extra={
                  budInfo &&
                  budInfo.hasCurrentDateData &&
                  budInfo.budCounts[currentViewAngle] > 0 && (
                    <Link
                      to={`/bud-analysis/${plantData.plantId}`}
                      state={{ date: selectedDate, images: images }}
                    >
                      Detailed Analysis
                    </Link>
                  )
                }
                style={{ background: "#e6feffee" }}
              >
                <Skeleton loading={loading} active>
                  {budInfo && (
                    <>
                      <Row gutter={16}>
                        <Col span={24}>
                          <Statistic
                            title={
                              <span style={{ fontSize: "14px" }}>
                                Number of Buds (Current View)
                              </span>
                            }
                            value={budInfo.budCounts[currentViewAngle] || 0}
                            prefix={<ExperimentOutlined />}
                            valueStyle={{ fontSize: "20px" }}
                          />
                        </Col>
                      </Row>
                      {/* <Divider style={{ margin: "16px 0" }} /> */}
                      {/* <Text type="secondary" style={{ fontSize: "14px" }}>
                        Current date: {selectedDate}
                        {budInfo.hasCurrentDateData
                          ? budInfo.budCounts[currentViewAngle] > 0
                            ? " (Buds detected)"
                            : " (No buds detected)"
                          : " (No data available)"}
                      </Text> */}
                    </>
                  )}
                </Skeleton>
              </Card>
              <Card
                title={
                  <span style={{ fontSize: "16px" }}>Branch Analysis</span>
                }
                extra={
                  branchInfo &&
                  branchInfo.hasCurrentDateData && (
                    <Link
                      to={`/branch-analysis/${plantData.plantId}`}
                      state={{ date: selectedDate, images: images }}
                    >
                      Detailed Analysis
                    </Link>
                  )
                }
                style={{ background: "#f6ffed" }}
              >
                <Skeleton loading={loading} active>
                  {branchInfo && (
                    <>
                      <Row gutter={[16, 16]}>
                        <Col span={12}>
                          <Statistic
                            title={
                              <span style={{ fontSize: "14px" }}>
                                Total Branches
                              </span>
                            }
                            value={branchInfo.totalBranches - 1}
                            prefix={<BranchesOutlined />}
                            valueStyle={{ fontSize: "20px" }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title={
                              <span style={{ fontSize: "14px" }}>
                                Main Stem Length
                              </span>
                            }
                            value={branchInfo.mainStemLength?branchInfo.mainStemLength:'0'}
                            suffix="pixels"
                            prefix={<BranchesOutlined />}
                            valueStyle={{ fontSize: "20px" }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title={
                              <span style={{ fontSize: "14px" }}>
                                Primary Branches
                              </span>
                            }
                            value={branchInfo.secondaryBranches}
                            prefix={<BranchesOutlined />}
                            valueStyle={{ fontSize: "20px" }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title={
                              <span style={{ fontSize: "14px" }}>
                                Secondary Branches
                              </span>
                            }
                            value={branchInfo.tertiaryBranches}
                            prefix={<BranchesOutlined />}
                            valueStyle={{ fontSize: "20px" }}
                          />
                        </Col>
                      </Row>
                      {/* <Divider style={{ margin: "16px 0" }} />
                      <Text type="secondary" style={{ fontSize: "14px" }}>
                        Earliest data date: {branchInfo.earliestDate}
                      </Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: "14px" }}>
                        Current date: {selectedDate}
                        {branchInfo.hasCurrentDateData
                          ? " (Data available)"
                          : " (No data available)"}
                      </Text> */}
                    </>
                  )}
                </Skeleton>
              </Card>
            </Space>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
}

export default withPageTransition(HomePage);
