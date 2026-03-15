/**
 * Layout Debug Tool - 打印 sidebar 和主内容区的位置关系
 * 使用方法：在浏览器控制台运行 window.layoutDebug.print()
 */
(function() {
    'use strict';

    const layoutDebug = {
        /**
         * 获取元素的计算样式
         */
        getElementInfo(selector) {
            const el = document.querySelector(selector);
            if (!el) {
                return { error: `元素未找到：${selector}` };
            }

            const rect = el.getBoundingClientRect();
            const styles = window.getComputedStyle(el);

            return {
                selector,
                element: el.tagName.toLowerCase(),
                classes: el.className,
                rect: {
                    top: rect.top.toFixed(2),
                    right: rect.right.toFixed(2),
                    bottom: rect.bottom.toFixed(2),
                    left: rect.left.toFixed(2),
                    width: rect.width.toFixed(2),
                    height: rect.height.toFixed(2)
                },
                position: styles.position,
                zIndex: styles.zIndex,
                display: styles.display,
                overflow: {
                    x: styles.overflowX,
                    y: styles.overflowY
                },
                margin: {
                    top: styles.marginTop,
                    right: styles.marginRight,
                    bottom: styles.marginBottom,
                    left: styles.marginLeft
                },
                padding: {
                    top: styles.paddingTop,
                    right: styles.paddingRight,
                    bottom: styles.paddingBottom,
                    left: styles.paddingLeft
                },
                border: {
                    right: styles.borderRight,
                    rightWidth: styles.borderRightWidth
                },
                transform: styles.transform,
                visibility: styles.visibility,
                opacity: styles.opacity
            };
        },

        /**
         * 打印重叠分析报告
         */
        analyzeOverlap() {
            const sidebar = this.getElementInfo('.sidebar');
            const homeContainer = this.getElementInfo('.home-container');
            const postsSection = this.getElementInfo('.posts-section');
            const siteHeader = this.getElementInfo('.site-header');

            console.group('📐 布局分析报告');
            
            // 窗口信息
            console.log('%c📱 窗口信息', 'font-weight: bold; color: #3498db;');
            console.table({
                '窗口宽度': window.innerWidth + 'px',
                '窗口高度': window.innerHeight + 'px',
                '设备像素比': window.devicePixelRatio
            });

            // 各元素位置
            console.log('%c📍 元素位置', 'font-weight: bold; color: #2ecc71;');
            console.table({
                'sidebar': sidebar.rect,
                'home-container': homeContainer.rect || 'N/A',
                'posts-section': postsSection.rect || 'N/A',
                'site-header': siteHeader.rect || 'N/A'
            });

            // 层级关系
            console.log('%c🏗️ Z-Index 层级', 'font-weight: bold; color: #e74c3c;');
            console.table({
                'sidebar': sidebar.zIndex,
                'site-header': siteHeader.zIndex,
                'theme-toggle': window.getComputedStyle(document.querySelector('#theme-toggle'))?.zIndex || 'N/A'
            });

            // 定位和溢出
            console.log('%c🔧 Position & Overflow', 'font-weight: bold; color: #f39c12;');
            console.table({
                'sidebar': {
                    position: sidebar.position,
                    overflowX: sidebar.overflow.x,
                    overflowY: sidebar.overflow.y,
                    transform: sidebar.transform
                },
                'home-container': homeContainer.position ? {
                    position: homeContainer.position,
                    overflowX: homeContainer.overflow?.x,
                    overflowY: homeContainer.overflow?.y
                } : 'N/A',
                'posts-section': postsSection.position ? {
                    position: postsSection.position,
                    overflowX: postsSection.overflow?.x,
                    overflowY: postsSection.overflow?.y
                } : 'N/A'
            });

            // 边距
            console.log('%c📏 Margin (Left)', 'font-weight: bold; color: #9b59b6;');
            console.table({
                'sidebar': sidebar.margin.left,
                'home-container': homeContainer.margin?.left || 'N/A',
                'posts-section': postsSection.margin?.left || 'N/A',
                'site-header': siteHeader.margin?.left || 'N/A'
            });

            // 重叠检测
            console.log('%c⚠️ 重叠检测', 'font-weight: bold; color: #e67e22;');
            
            if (sidebar.rect && homeContainer.rect) {
                const sidebarRight = parseFloat(sidebar.rect.right);
                const homeLeft = parseFloat(homeContainer.rect.left);
                
                const overlap = sidebarRight > homeLeft;
                const gap = homeLeft - sidebarRight;
                
                console.log(`Sidebar 右边界：${sidebarRight}px`);
                console.log(`Home-container 左边界：${homeLeft}px`);
                console.log(`间距/重叠：${gap > 0 ? gap + 'px (间距)' : Math.abs(gap) + 'px (重叠)'}`);
                
                if (overlap) {
                    console.warn('⚠️ 检测到重叠！sidebar 覆盖在 home-container 上方');
                } else {
                    console.log('✅ 无重叠');
                }
            }

            // Sidebar 激活状态
            console.log('%c🔌 Sidebar 状态', 'font-weight: bold; color: #1abc9c;');
            const sidebarEl = document.querySelector('.sidebar');
            const bodyEl = document.body;
            console.table({
                'sidebar.active': sidebarEl?.classList.contains('active'),
                'body.sidebar-active': bodyEl?.classList.contains('sidebar-active'),
                'sidebar visible': sidebarEl && sidebar.transform !== 'translateX(-100%)'
            });

            console.groupEnd();
        },

        /**
         * 打印简化的位置关系（快速查看）
         */
        quickView() {
            const sidebar = document.querySelector('.sidebar');
            const homeContainer = document.querySelector('.home-container');
            
            if (!sidebar || !homeContainer) {
                console.warn('元素未找到');
                return;
            }

            const sidebarRect = sidebar.getBoundingClientRect();
            const homeRect = homeContainer.getBoundingClientRect();
            const styles = window.getComputedStyle(sidebar);

            console.log('%c╔════════════════════════════════════════════╗', 'color: #7f8c8d;');
            console.log('%c║         📐 Layout Quick View              ║', 'color: #3498db; font-weight: bold;');
            console.log('%c╚════════════════════════════════════════════╝', 'color: #7f8c8d;');
            console.log('');
            console.log(`窗口宽度：${window.innerWidth}px`);
            console.log(`Sidebar: [${sidebarRect.left.toFixed(0)}, ${sidebarRect.right.toFixed(0)}] transform=${styles.transform}`);
            console.log(`Home:    [${homeRect.left.toFixed(0)}, ${homeRect.right.toFixed(0)}]`);
            console.log(`间距：${(homeRect.left - sidebarRect.right).toFixed(0)}px ${homeRect.left - sidebarRect.right > 0 ? '✅' : '⚠️ 重叠'}`);
            console.log(`Sidebar active: ${sidebar.classList.contains('active')}`);
            console.log(`Body sidebar-active: ${document.body.classList.contains('sidebar-active')}`);
        },

        /**
         * 监听窗口变化
         */
        watch() {
            console.log('👁️ 开始监听窗口变化... (运行 window.layoutDebug.unwatch() 停止)');
            
            let timeout;
            const handler = () => {
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    this.quickView();
                }, 100);
            };

            window.addEventListener('resize', handler);
            this._watchHandler = handler;
        },

        unwatch() {
            if (this._watchHandler) {
                window.removeEventListener('resize', this._watchHandler);
                console.log('❌ 停止监听');
            }
        },

        /**
         * 主打印方法
         */
        print() {
            console.clear();
            console.log('%c╔══════════════════════════════════════════════════════════╗', 'color: #7f8c8d;');
            console.log('%c║           🎨 Layout Debug Tool - 布局调试工具            ║', 'color: #3498db; font-weight: bold; font-size: 14px;');
            console.log('%c╚══════════════════════════════════════════════════════════╝', 'color: #7f8c8d;');
            console.log('');
            console.log('可用命令:');
            console.log('  window.layoutDebug.print()    - 完整分析报告');
            console.log('  window.layoutDebug.quickView() - 快速查看');
            console.log('  window.layoutDebug.watch()     - 监听窗口变化');
            console.log('  window.layoutDebug.unwatch()   - 停止监听');
            console.log('');
            this.analyzeOverlap();
        }
    };

    // 暴露到全局
    window.layoutDebug = layoutDebug;

    // 自动运行（仅在开发环境）
    if (window.location.hostname === 'localhost' || window.location.port) {
        console.log('💡 运行 window.layoutDebug.print() 查看布局分析');
    }
})();
